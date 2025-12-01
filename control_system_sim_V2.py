import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
import csv

# Functions from python control package
tau = 0.45
G_servo = ct.tf([1], [tau, 1])  # first order: 1 / (tau s + 1)
# G_servo = ct.tf([1], [tau, 1, 0])  # 1 / (tau s^2 + s)


# ============================================================
# Servo datasheet based constants and conversions
# ============================================================

SERVO_GEAR_RATIO = 7.0           # 7:1 from datasheet (not used directly here)
SERVO_VOLTAGE_MIN = 6.0
SERVO_VOLTAGE_MAX = 7.4

# No load speeds from datasheet, converted to deg/s
SERVO_SPEED_6V_DEG_PER_S = 60.0 / 1.19   # 1.19 s per 60 deg
SERVO_SPEED_7_4V_DEG_PER_S = 60.0 / 0.98 # 0.98 s per 60 deg

# PWM and travel info
PWM_MIN_US = 800.0
PWM_MAX_US = 2200.0
PWM_CENTER_US = (PWM_MIN_US + PWM_MAX_US) / 2.0

TRAVEL_DEG_PER_US = 0.26
ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 360.0
ANGLE_CENTER_DEG = 180.0

PWM_DEADBAND_US = 1.0   # deadband width from datasheet


def servo_speed_from_voltage(voltage):
    """
    Interpolate no load speed between 6 V and 7.4 V based on datasheet.
    Returns max speed in deg/s.
    """
    v = float(voltage)
    if v <= SERVO_VOLTAGE_MIN:
        return SERVO_SPEED_6V_DEG_PER_S
    if v >= SERVO_VOLTAGE_MAX:
        return SERVO_SPEED_7_4V_DEG_PER_S
    frac = (v - SERVO_VOLTAGE_MIN) / (SERVO_VOLTAGE_MAX - SERVO_VOLTAGE_MIN)
    return SERVO_SPEED_6V_DEG_PER_S + frac * (SERVO_SPEED_7_4V_DEG_PER_S - SERVO_SPEED_6V_DEG_PER_S)


def pwm_to_angle_deg(pulse_us):
    """
    Map PWM pulse width to absolute angle using:
      travel per us = 0.26 deg/us
      800-2200 us -> about 0-360 deg
      1500 us -> about 180 deg
    Then clamp to [0, 360].
    """
    pulse_us = np.clip(pulse_us, PWM_MIN_US, PWM_MAX_US)
    angle = ANGLE_CENTER_DEG + (pulse_us - PWM_CENTER_US) * TRAVEL_DEG_PER_US
    return float(np.clip(angle, ANGLE_MIN_DEG, ANGLE_MAX_DEG))


def angle_deg_to_pwm(angle_deg):
    """
    Inverse mapping of pwm_to_angle_deg:
      0-360 deg -> 800-2200 us (approximately)
    """
    angle_deg = np.clip(angle_deg, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    pulse = PWM_CENTER_US + (angle_deg - ANGLE_CENTER_DEG) / TRAVEL_DEG_PER_US
    return float(np.clip(pulse, PWM_MIN_US, PWM_MAX_US))


def make_pid_tf(kp, ki, kd):
    """
    PID controller tf is C(s) = (Kd*s^2 + Kp*s + Ki) / s
    """
    num = [kd, kp, ki]
    den = [1, 0]
    return ct.tf(num, den)


def make_closed_loop(kp, ki, kd):
    """
    Closed-loop tf is T(s) = feedback(C(s)*G(s), 1)
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


def pid_step_cost(
    t,
    y,
    beta=0.2,
    gamma=3.0,
    delta=10.0,
    target=1.0,
    tol=0.02,
):
    """
    J = IAE + beta * Ts + gamma * Mp + delta * |e_ss|

    where:
      IAE  = integral of absolute error
      Ts   = settling time
      Mp   = overshoot (absolute, not percent)
      e_ss = steady state error (last sample)
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if len(t) < 2:
        return np.inf

    dt = t[1] - t[0]

    # error for unit step
    e = target - y

    # integral of absolute error
    iae = np.sum(np.abs(e)) * dt

    # settling time
    Ts = settling_time_from_response(t, y, tol=tol)

    # overshoot
    Mp = max(0.0, np.max(y) - target)

    # steady state error (last sample)
    e_ss = abs(e[-1])

    # total cost
    J = iae + beta * Ts + gamma * Mp + delta * e_ss
    return J


def tune_pid():
    """
    Simple grid search over PID gains using python-control on the first-order servo model.
    Minimizes a cost combining settling time, overshoot, and steady state error.
    """
    kp_vals = np.linspace(0.5, 8.0, 12)
    ki_vals = np.linspace(0.0, 5.0, 11)
    kd_vals = np.linspace(0.0, 1.0, 6)

    best_cost = 1e9
    best_pid = None

    t = np.linspace(0, 5, 500)  # 5 s step response

    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                try:
                    T_cl = make_closed_loop(kp, ki, kd)
                    t_out, y_out = ct.step_response(T_cl, T=t)
                except Exception:
                    continue

                cost = pid_step_cost(
                    t_out,
                    y_out,
                    beta=0.2,
                    gamma=3.0,
                    delta=10.0,
                    target=1.0,
                    tol=0.02,
                )

                if cost < best_cost:
                    best_cost = cost
                    best_pid = (kp, ki, kd)
                    print(
                        f"New best (continuous model): "
                        f"cost={best_cost:.3f}, kp={kp:.3f}, "
                        f"ki={ki:.3f}, kd={kd:.3f}"
                    )

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
    First order position servo with PWM input, voltage dependent speed,
    and 0-360 deg travel.

    Internally:
      angle[k+1] = angle[k] + rate * dt
      rate ~ (cmd_angle - angle) / tau, limited to +/- rate_limit
    """
    def __init__(self, init_angle=180.0, tau=0.45, voltage=7.4):
        # clamp initial angle to valid range
        self.angle = float(np.clip(init_angle, ANGLE_MIN_DEG, ANGLE_MAX_DEG))
        self.tau = float(tau)
        self.voltage = float(voltage)
        self.rate_limit = servo_speed_from_voltage(self.voltage)  # deg/s
        self.last_pwm = PWM_CENTER_US

    def set_voltage(self, voltage):
        """
        Change supply voltage at runtime and update speed limit.
        """
        self.voltage = float(voltage)
        self.rate_limit = servo_speed_from_voltage(self.voltage)

    def step_angle(self, cmd_angle_deg, dt):
        """
        Direct angle domain step (bypasses PWM mapping).
        Still respects voltage based rate limit and 0-360 deg clamp.
        """
        cmd = float(np.clip(cmd_angle_deg, ANGLE_MIN_DEG, ANGLE_MAX_DEG))
        raw_rate = (cmd - self.angle) / max(self.tau, 1e-6)
        raw_rate = np.clip(raw_rate, -self.rate_limit, self.rate_limit)
        self.angle = float(
            np.clip(self.angle + raw_rate * dt, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        )
        return self.angle

    def step_pwm(self, pulse_us, dt):
        """
        Step using PWM input in microseconds, including deadband and travel per us.
        """
        raw_pwm = float(pulse_us)

        # Apply deadband in PWM domain: changes smaller than 1 us produce no motion
        if abs(raw_pwm - self.last_pwm) <= PWM_DEADBAND_US:
            effective_pwm = self.last_pwm
        else:
            effective_pwm = raw_pwm

        self.last_pwm = raw_pwm

        # Convert PWM to commanded angle
        cmd_angle = pwm_to_angle_deg(effective_pwm)

        # First order dynamics with rate limit based on supply voltage
        raw_rate = (cmd_angle - self.angle) / max(self.tau, 1e-6)
        raw_rate = np.clip(raw_rate, -self.rate_limit, self.rate_limit)

        self.angle = float(
            np.clip(self.angle + raw_rate * dt, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        )
        return self.angle


def get_command(t_now, time_cmd, cmd_values):
    """
    Interpolate command value to minimize jagged responsiveness
    """
    return np.interp(t_now, time_cmd, cmd_values)


def kalman_1d_angle(time, meas, q=0.01, r=1.0):
    """
    1D Kalman filter to help minimize jagged responsiveness.
    q and r are hyperparameters that we can continue to tune.
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

        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
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

    # Write CSV
    filename = "sample_commands.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "az_deg", "el_deg"])
        writer.writerows(data)

    print(f"Wrote {samples} samples to {filename}")


def simulate(
    kp=0.8,
    ki=0.02,
    kd=0.05,
    dt=0.05,
    sim_time=10.0,
    cmd_file='sample_commands_filtered.csv',
    angle_limits=(ANGLE_MIN_DEG, ANGLE_MAX_DEG),
    dcmd_limit_per_step=5.0,  # currently unused but kept as parameter
    plot=True,
    return_logs=False,
    servo_voltage=7.4,
):

    cmd_df = pd.read_csv(cmd_file)
    time_cmd = cmd_df['time'].values
    az_cmd = cmd_df['az_deg_filt'].values
    el_cmd = cmd_df['el_deg_filt'].values

    # Start servos at first commanded angles
    init_az = float(az_cmd[0])
    init_el = float(el_cmd[0])

    servo_pan = FirstOrderServo(init_angle=init_az, tau=tau, voltage=servo_voltage)
    servo_tilt = FirstOrderServo(init_angle=init_el, tau=tau, voltage=servo_voltage)

    # PID states
    prev_err_az, prev_err_el = 0.0, 0.0
    integral_az, integral_el = 0.0, 0.0

    # Logs
    time_log = []
    az_log, el_log = [], []
    az_err_log, el_err_log = [], []
    sp_az_log, sp_el_log = [], []

    # Commanded states (servo setpoints)
    az_cmd_state = init_az
    el_cmd_state = init_el

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

        # Update error memory
        prev_err_az, prev_err_el = az_err, el_err

        # Treat PID output as an absolute angle command
        az_cmd_state = np.clip(az_control, angle_limits[0], angle_limits[1])
        el_cmd_state = np.clip(el_control, angle_limits[0], angle_limits[1])

        # Convert angle command to PWM, then step servo using PWM
        pwm_az = angle_deg_to_pwm(az_cmd_state)
        pwm_el = angle_deg_to_pwm(el_cmd_state)

        new_az = servo_pan.step_pwm(pwm_az, dt)
        new_el = servo_tilt.step_pwm(pwm_el, dt)

        # Log for graph
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
        plt.savefig("mags_sim_plot_v2.png")
        print("Saved plot to mags_sim_plot_v2.png")

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


# ============================================================
# STAGE 2: TRAJECTORY-BASED COST AND LOCAL TUNING (TEST ONLY NOT ACTUALLY USED)
# ============================================================

def trajectory_cost(kp, ki, kd,
                    dt=0.05,
                    sim_time=10.0,
                    cmd_file='sample_commands_filtered.csv',
                    big_err_deg=5.0):
    """
    Cost based on actual trajectory tracking.

    J_traj = IAE_traj + 10 * T_big_err

    where:
      IAE_traj   = integral of |e_az| + |e_el|
      T_big_err  = time with max(|e_az|,|e_el|) > big_err_deg
    """
    logs = simulate(
        kp=kp,
        ki=ki,
        kd=kd,
        dt=dt,
        sim_time=sim_time,
        cmd_file=cmd_file,
        plot=False,
        return_logs=True,
    )

    if logs is None:
        return np.inf

    t = logs["time"]
    if len(t) < 2:
        return np.inf

    az = logs["az"]
    el = logs["el"]
    sp_az = logs["sp_az"]
    sp_el = logs["sp_el"]

    e_az = sp_az - az
    e_el = sp_el - el

    dt = t[1] - t[0]

    iae_traj = np.sum(np.abs(e_az) + np.abs(e_el)) * dt

    max_err = np.maximum(np.abs(e_az), np.abs(e_el))
    big_mask = max_err > big_err_deg
    big_time = np.sum(big_mask) * dt

    cost = iae_traj + 10.0 * big_time
    return cost


def tune_pid_trajectory(base_kp, base_ki, base_kd):
    """
    Stage 2:
    Local search around the step-tuned PID using trajectory cost.
    """

    # build local ranges around the base gains
    kp_vals = np.linspace(0.7 * base_kp, 1.3 * base_kp, 7)
    ki_vals = np.linspace(0.7 * base_ki, 1.3 * base_ki, 7)
    kd_vals = np.linspace(0.0, 0.5 * max(0.1, base_kp), 5)  # small Kd range

    best_cost = np.inf
    best_pid = (base_kp, base_ki, base_kd)

    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                cost = trajectory_cost(
                    kp, ki, kd,
                    dt=0.05,
                    sim_time=10.0,
                    cmd_file='sample_commands_filtered.csv',
                    big_err_deg=5.0,
                )

                if not np.isfinite(cost):
                    continue

                if cost < best_cost:
                    best_cost = cost
                    best_pid = (kp, ki, kd)
                    print(
                        f"New best (traj cost): "
                        f"cost={best_cost:.3f}, kp={kp:.3f}, "
                        f"ki={ki:.3f}, kd={kd:.3f}"
                    )

    print("\nBest PID from trajectory tuning:", best_pid)
    return best_pid

def summarize_tracking(logs, big_err_deg=5.0):
    t = logs["time"]
    e_az = logs["sp_az"] - logs["az"]
    e_el = logs["sp_el"] - logs["el"]

    dt = t[1] - t[0]

    max_err_az = np.max(np.abs(e_az))
    max_err_el = np.max(np.abs(e_el))

    iae = np.sum(np.abs(e_az) + np.abs(e_el)) * dt

    max_err = np.maximum(np.abs(e_az), np.abs(e_el))
    big_time = np.sum(max_err > big_err_deg) * dt

    print(f"Max az error  : {max_err_az:.2f} deg")
    print(f"Max el error  : {max_err_el:.2f} deg")
    print(f"IAE (total)   : {iae:.2f} deg·s")
    print(f"Time |err|>{big_err_deg}°: {big_time:.2f} s")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    fs = 10.0       # 10 Hz sampling
    samples = 500   # 500 time steps
    generate_samples_commands(fs, samples)

    

    apply_kalman_to_commands(
        in_file='sample_commands.csv',
        out_file='sample_commands_filtered.csv',
        q=0.01,
        r=1.0,
    )

    print("Tuning PID on continuous first-order servo model")
    best_kp, best_ki, best_kd = tune_pid()

    # base_kp, base_ki, base_kd = tune_pid() 

    # best_kp, best_ki, best_kd = tune_pid_trajectory(base_kp, base_ki, base_kd)

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
        servo_voltage=7.4,
    )

    print("Done.")