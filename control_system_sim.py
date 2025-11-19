import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
import csv

# Functions from python control package
tau = 0.15  # this tau can be tuned 
G_servo = ct.tf([1], [tau, 1])  # 1 / (tau s + 1)

def make_pid_tf(kp, ki, kd):
    """
    PID controller tf is C(s) = (Kd*s^2 + Kp*s + Ki) / s
    """
    num = [kd, kp, ki]
    den = [1, 0]   
    return ct.tf(num, den)

def make_closed_loop(kp, ki, kd):
    """
    Closed-loop tf is  T(s) = feedback(C(s)*G(s), 1)
    """
    
    C = make_pid_tf(kp, ki, kd)
    L = C * G_servo          # open-loop
    T = ct.feedback(L, 1)    # unity feedback
    return T

def settling_time_from_response(t, y, tol=0.02):
    y_err = np.abs(y - 1.0)
    for i in range(len(t)):
        if np.all(y_err[i:] < tol):
            return t[i]
    return t[-1]

def tune_pid():
    """
    simple grid search over PID gains using python-control on the first-order servo model.
    Minimizes a cost combining settling time + overshoot.
    """
    kp_vals = np.linspace(0.2, 2.0, 10)
    ki_vals = np.linspace(0.0, 0.5, 6)
    kd_vals = np.linspace(0.0, 0.3, 7)

    best_cost = 1e9
    best_pid = None

    t = np.linspace(0, 5, 500)  # 5 s step response

    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                T_cl = make_closed_loop(kp, ki, kd)
                t_out, y_out = ct.step_response(T_cl, T=t)

                st = settling_time_from_response(t_out, y_out, tol=0.02)
                overshoot = max(0.0, np.max(y_out) - 1.0)  # only positive overshoot

                cost = st + 0.1 * overshoot  # tradeoff; tweak weighting as you like

                if cost < best_cost:
                    best_cost = cost
                    best_pid = (kp, ki, kd)
                    print(f"New best (continuous model): "
                          f"cost={best_cost:.3f}, kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}")

    print("\nBest PID:", best_pid)
    return best_pid

def plot_continuous_step_response(kp, ki, kd):
    """
    Plot step response of the continuous-time closed-loop model
    for the chosen PID gains.
    """
    T_cl = make_closed_loop(kp, ki, kd)
    t = np.linspace(0, 5, 500)
    t_out, y_out = ct.step_response(T_cl, T=t)

    plt.figure()
    plt.plot(t_out, y_out, label='Closed-loop response')
    plt.axhline(1.0, linestyle='--', label='Step (1.0)')
    plt.xlabel('Time (s)')
    plt.ylabel('Output (normalized angle)')
    plt.title(f'Continuous-time step response (kp={kp:.2f}, ki={ki:.2f}, kd={kd:.2f})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt):
    """
    Discrete PID controller for simulation.
    """
    error = setpoint - pv
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, error, integral


class FirstOrderServo:
    """
    angle[k+1] = angle[k] + rate * dt
    where rate ~ (cmd - angle) / tau
    """
    def __init__(self, init_angle=0.0, tau=0.15, rate_limit=180.0):
        self.angle = float(init_angle)
        self.tau = float(tau)
        self.rate_limit = rate_limit

    def step(self, cmd_deg, dt):
        raw_rate = (cmd_deg - self.angle) / max(self.tau, 1e-6)
        if self.rate_limit is not None:
            raw_rate = np.clip(raw_rate, -self.rate_limit, self.rate_limit)
        self.angle += raw_rate * dt
        return self.angle
    
def get_command(t_now, time_cmd, cmd_values):
    """
    Interpolate command value to minimize jagged responsiveness
    """
    return np.interp(t_now, time_cmd, cmd_values)

def kalman_1d_angle(time, meas, q=0.01, r=1.0):
    """
    1D Kalman filter once again to help minimize jagged responsiveness
    q and r are hyperparameters that we can continue to tune
    """
    time = np.asarray(time)
    meas = np.asarray(meas)
    n = len(time)

    # state history angle, rate
    x_hist = np.zeros((n, 2))

    # initial state
    x = np.array([meas[0], 0.0])

    # initial covariance
    P = np.eye(2) * 1.0

    # measurement matrix and noise
    H = np.array([[1.0, 0.0]]) 
    R = np.array([[r]])

    x_hist[0] = x

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        if dt <= 0:
            dt = 1e-3  

        # State transition matrix 
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Process noise 
        Q = q * np.array([[dt**4 / 4.0, dt**3 / 2.0],
                          [dt**3 / 2.0, dt**2]])

        # --- Prediction ---
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # --- Update ---
        z = np.array([[meas[k]]])  # measurement at this step
        y = z - H @ x_pred         # innovation
        S = H @ P_pred @ H.T + R   # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x_pred + (K @ y).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

        x_hist[k] = x

    angles = x_hist[:, 0]
    rates = x_hist[:, 1]
    return angles, rates

def apply_kalman_to_commands(in_file='sample_commands.csv',
                             out_file='sample_commands_filtered.csv',
                             q=0.01,
                             r=1.0):
    # Load original commands
    df = pd.read_csv(in_file)

    t = df['time'].values
    az = df['az_deg'].values
    el = df['el_deg'].values

    # Run Kalman filter independently on az and el
    az_filt, az_rate = kalman_1d_angle(t, az, q=q, r=r)
    el_filt, el_rate = kalman_1d_angle(t, el, q=q, r=r)

    # Store back into dataframe
    df['az_deg_filt'] = az_filt
    df['el_deg_filt'] = el_filt
    df['az_rate_filt'] = az_rate
    df['el_rate_filt'] = el_rate

    # Save to new CSV
    df.to_csv(out_file, index=False)
    print(f"Filtered commands written to {out_file}")


def generate_samples_commands(fs, samples):
    """
    Kinematics:
      Horizontal:
        x(t) = 160 - t
        y(t) = t

      Vertical (z):
        z(t) = -2.8 t (t - 155)          for 0 < t < 20
        z'(t) = -54 t + 8640             for 20 <= t < 160
    """

    # Time vector
    t = np.arange(samples, dtype=float) / float(fs)

    # z(t) (piecewise)
    def z_of_t(t_arr):
        t_arr = np.asarray(t_arr, dtype=float)
        z = np.zeros_like(t_arr)

        # Region 0 < t < 20
        mask1 = (t_arr > 0) & (t_arr < 20)
        z[mask1] = -2.8 * t_arr[mask1] * (t_arr[mask1] - 155.0)

        # Region 20 <= t < 160
      
        mask2 = (t_arr >= 20)
        z[mask2] = -54.0 * t_arr[mask2] + 8640.0

        return z

    # Horizontal 
    x = 160.0 - t
    y = t
    z = z_of_t(t)

    # Convert to spherical angles  
  
    az_rad = np.arctan2(y, x)
    az_deg = np.degrees(az_rad)
    az_deg = (az_deg + 360.0) % 360.0


    r_xy = np.sqrt(x**2 + y**2)
    el_rad = np.arctan2(z, r_xy)
    el_deg = np.degrees(el_rad)

    data = np.column_stack([t, az_deg, el_deg])

    # --- Write CSV ---
    filename = "sample_commands.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "az_deg", "el_deg"])
        writer.writerows(data)

    print(f"Wrote {samples} samples to {filename}")


def simulate(kp=0.8, ki=0.02, kd=0.05,
             dt=0.05, sim_time=10.0,
             cmd_file='sample_commands.csv',
             angle_limits=(-90, 90),
             dcmd_limit_per_step=5.0,
             plot=True,
             return_logs=False):

    

    cmd_df = pd.read_csv(cmd_file)
    time_cmd = cmd_df['time'].values
    az_cmd   = cmd_df['az_deg_filt'].values
    el_cmd   = cmd_df['el_deg_filt'].values


    # Initialize servos
    servo_pan = FirstOrderServo(init_angle=0.0, tau=tau)
    servo_tilt = FirstOrderServo(init_angle=0.0, tau=tau)

    # PID states
    prev_err_az, prev_err_el = 0.0, 0.0
    integral_az, integral_el = 0.0, 0.0

    # Logs
    time_log = []
    az_log, el_log = [], []
    az_err_log, el_err_log = [], []
    sp_az_log, sp_el_log = [], []

    # Commanded states (servo setpoints)
    az_cmd_state = servo_pan.angle
    el_cmd_state = servo_tilt.angle

    t_vals = np.arange(0.0, sim_time, dt)
    for t in t_vals:
        curr_az = servo_pan.angle
        curr_el = servo_tilt.angle

        # Interpolate commanded az/el from file at current time
        sp_az = get_command(t, time_cmd, az_cmd)
        sp_el = get_command(t, time_cmd, el_cmd)

        # PID control
        az_control, az_err, integral_az = pid_controller(
            sp_az, curr_az, kp, ki, kd, prev_err_az, integral_az, dt
        )
        el_control, el_err, integral_el = pid_controller(
            sp_el, curr_el, kp, ki, kd, prev_err_el, integral_el, dt
        )

        # Update error mem
        prev_err_az, prev_err_el = az_err, el_err

        # Limit increments in commanded angle
        az_delta = np.clip(az_control, -dcmd_limit_per_step, dcmd_limit_per_step)
        el_delta = np.clip(el_control, -dcmd_limit_per_step, dcmd_limit_per_step)

        # Update commanded angles
        az_cmd_state = np.clip(az_cmd_state + az_delta, angle_limits[0], angle_limits[1])
        el_cmd_state = np.clip(el_cmd_state + el_delta, angle_limits[0], angle_limits[1])

        # Servo step
        new_az = servo_pan.step(az_cmd_state, dt)
        new_el = servo_tilt.step(el_cmd_state, dt)

        # Log (for graph)
        time_log.append(t)
        az_log.append(new_az)
        el_log.append(new_el)
        az_err_log.append(az_err)
        el_err_log.append(el_err)
        sp_az_log.append(sp_az)
        sp_el_log.append(sp_el)

    # Convert to arrays
    time_log = np.asarray(time_log)
    az_log = np.asarray(az_log)
    el_log = np.asarray(el_log)
    az_err_log = np.asarray(az_err_log)
    el_err_log = np.asarray(el_err_log)
    sp_az_log = np.asarray(sp_az_log)
    sp_el_log = np.asarray(sp_el_log)

    if plot:
        plt.figure(figsize=(10, 6))

        # Angles vs time
        plt.subplot(2, 1, 1)
        plt.plot(time_log, az_log, label='Current Azimuth (°)')
        plt.plot(time_log, el_log, label='Current Elevation (°)')

        # Plot commanded signals
        plt.plot(time_log, sp_az_log, '--', label='Commanded Azimuth (°)')
        plt.plot(time_log, sp_el_log, '--', label='Commanded Elevation (°)')

        plt.title('Discrete Servo Angles vs Time (MAGS sim)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid(True)

        # Errors vs time
        plt.subplot(2, 1, 2)
        plt.plot(time_log, az_err_log, label='Azimuth Error (°)')
        plt.plot(time_log, el_err_log, label='Elevation Error (°)')
        plt.title('Discrete Tracking Error vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (°)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("mags_sim_plot.png")
        print("Saved plot to mags_sim_plot.png")
        # plt.show()  # comment this for now


    if return_logs:
        return {
            "time": time_log,
            "az": az_log,
            "el": el_log,
            "az_err": az_err_log,
            "el_err": el_err_log,
            "sp_az": sp_az_log,
            "sp_el": sp_el_log,
        }

    return None



if __name__ == "__main__":
    fs = 10.0       # 10 Hz sampling
    samples = 500   # 500 time steps
    generate_samples_commands(fs, samples)

    print("Tuning PID on continuous first-order servo model")
    best_kp, best_ki, best_kd = tune_pid()

    apply_kalman_to_commands(
        in_file='sample_commands.csv',
        out_file='sample_commands_filtered.csv',
        q=0.01,
        r=1.0,
    )

    print("Starting simulate...")
    simulate(
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
        dt=0.05,
        sim_time=10.0,
        cmd_file='sample_commands_filtered.csv',
        plot=True,
        return_logs=False,
    )
    print("Done.")


