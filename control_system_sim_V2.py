import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
import csv

# ============================================================
# GLOBAL SERVO CONSTANTS
# ============================================================

tau_default = 0.45
G_servo = ct.tf([1], [tau_default, 1])  # first-order continuous model

SERVO_GEAR_RATIO = 7.0           # 7:1 from datasheet
SERVO_VOLTAGE_MIN = 6.0
SERVO_VOLTAGE_MAX = 7.4

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


# ============================================================
# HELPER FUNCTIONS: SERVO CHARACTERISTICS
# ============================================================

def servo_speed_from_voltage(voltage):
    """
    Interpolate no-load speed between 6 V and 7.4 V based on datasheet.
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
      0-360 deg -> 800-2200 us (approximately).
    """
    angle_deg = np.clip(angle_deg, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    pulse = PWM_CENTER_US + (angle_deg - ANGLE_CENTER_DEG) / TRAVEL_DEG_PER_US
    return float(np.clip(pulse, PWM_MIN_US, PWM_MAX_US))


# ============================================================
# PID TRANSFER FUNCTION + CONTINUOUS TUNING
# ============================================================

def make_pid_tf(kp, ki, kd):
    """
    Continuous-time PID controller: C(s) = (Kd*s^2 + Kp*s + Ki) / s
    """
    num = [kd, kp, ki]
    den = [1, 0]
    return ct.tf(num, den)


def make_closed_loop(kp, ki, kd, G=None):
    """
    Closed-loop continuous transfer function with unity feedback.
    """
    if G is None:
        G = G_servo
    C = make_pid_tf(kp, ki, kd)
    L = C * G
    T = ct.feedback(L, 1)
    return T


def settling_time_from_response(t, y, tol=0.02, target=1.0):
    """
    Settling time: earliest time after which |y - target| < tol for all remaining samples.
    """
    y_err = np.abs(y - target)
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
      Mp   = overshoot (absolute)
      e_ss = steady state error
    """
    t = np.asarray(t)
    y = np.asarray(y)
    if len(t) < 2:
        return np.inf

    dt = t[1] - t[0]
    e = target - y

    iae = np.sum(np.abs(e)) * dt
    Ts = settling_time_from_response(t, y, tol=tol, target=target)
    Mp = max(0.0, np.max(y) - target)
    e_ss = abs(e[-1])

    J = iae + beta * Ts + gamma * Mp + delta * e_ss
    return J


def tune_pid_step(G=None):
    """
    Grid-search PID gains on continuous-time first-order model using step cost.
    Returns best (kp, ki, kd).
    """
    if G is None:
        G = G_servo

    kp_vals = np.linspace(0.5, 8.0, 12)
    ki_vals = np.linspace(0.0, 5.0, 11)
    kd_vals = np.linspace(0.0, 1.0, 6)

    best_cost = np.inf
    best_pid = None

    t = np.linspace(0, 5, 500)  # 5 s step

    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                try:
                    T_cl = make_closed_loop(kp, ki, kd, G=G)
                    t_out, y_out = ct.step_response(T_cl, T=t)
                except Exception:
                    continue

                cost = pid_step_cost(t_out, y_out)

                if cost < best_cost:
                    best_cost = cost
                    best_pid = (kp, ki, kd)
                    print(
                        f"New best (step tuning): "
                        f"cost={best_cost:.3f}, kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}"
                    )

    print("\nBest PID from step tuning:", best_pid)
    return best_pid


# ============================================================
# DISCRETE PID + SERVO MODEL
# ============================================================

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
    First-order position servo with PWM input, voltage-dependent speed,
    and 0-360 deg travel.

    angle[k+1] = angle[k] + rate * dt
    rate ~ (cmd_angle - angle) / tau, limited to +/- rate_limit
    """

    def __init__(self, init_angle=180.0, tau=0.45, voltage=7.4):
        self.angle = float(np.clip(init_angle, ANGLE_MIN_DEG, ANGLE_MAX_DEG))
        self.tau = float(tau)
        self.voltage = float(voltage)
        self.rate_limit = servo_speed_from_voltage(self.voltage)  # deg/s
        self.last_pwm = PWM_CENTER_US

    def set_voltage(self, voltage):
        self.voltage = float(voltage)
        self.rate_limit = servo_speed_from_voltage(self.voltage)

    def set_tau(self, tau):
        self.tau = float(tau)

    def step_pwm(self, pulse_us, dt):
        """
        Step using PWM input in microseconds, including deadband and travel per us.
        """
        raw_pwm = float(pulse_us)

        if abs(raw_pwm - self.last_pwm) <= PWM_DEADBAND_US:
            effective_pwm = self.last_pwm
        else:
            effective_pwm = raw_pwm

        self.last_pwm = raw_pwm

        cmd_angle = pwm_to_angle_deg(effective_pwm)

        raw_rate = (cmd_angle - self.angle) / max(self.tau, 1e-6)
        raw_rate = np.clip(raw_rate, -self.rate_limit, self.rate_limit)

        self.angle = float(
            np.clip(self.angle + raw_rate * dt, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        )
        return self.angle


# ============================================================
# KALMAN FILTER FOR COMMAND SMOOTHING
# ============================================================

def kalman_1d_angle(time, meas, q=0.01, r=1.0):
    """
    1D constant-velocity Kalman filter for smoother angle commands.
    State: [angle, rate].
    """
    time = np.asarray(time)
    meas = np.asarray(meas)
    n = len(time)

    x_hist = np.zeros((n, 2))
    x = np.array([meas[0], 0.0])
    P = np.eye(2) * 1.0

    H = np.array([[1.0, 0.0]])   # measure angle only
    R = np.array([[r]])

    x_hist[0] = x

    for k in range(1, n):
        dt = time[k] - time[k - 1]
        if dt <= 0:
            dt = 1e-3

        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        Q = q * np.array([[dt**4 / 4.0, dt**3 / 2.0],
                          [dt**3 / 2.0, dt**2]])

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update
        z = np.array([[meas[k]]])
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + (K @ y).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

        x_hist[k] = x

    angles = x_hist[:, 0]
    rates = x_hist[:, 1]
    return angles, rates


# ============================================================
# COMMAND GENERATION + KALMAN APPLICATION
# ============================================================

def generate_samples_commands(fs, samples, filename="sample_commands.csv"):
    """
    Generate synthetic rocket trajectory and write to CSV.

    Kinematics:
      Horizontal:
        x(t) = 160 - t
        y(t) = t

      Vertical:
      
    """
    t = np.arange(samples, dtype=float) / float(fs)

    def z_of_t(t_arr):
        t_arr = np.asarray(t_arr, dtype=float)
        z = np.zeros_like(t_arr)

        mask1 = (t_arr > 0) & (t_arr < 20)
        z[mask1] = -2.8 * t_arr[mask1] * (t_arr[mask1] - 155.0)

        mask2 = (t_arr >= 20)
        z[mask2] = -54.0 * t_arr[mask2] + 8640.0

        return z

    x = 160.0 - t
    y = t
    z = z_of_t(t)

    az_rad = np.arctan2(y, x)
    az_deg = np.degrees(az_rad)
    az_deg = (az_deg + 360.0) % 360.0

    r_xy = np.sqrt(x**2 + y**2)
    el_rad = np.arctan2(z, r_xy)
    el_deg = np.degrees(el_rad)

    data = np.column_stack([t, az_deg, el_deg])

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "az_deg", "el_deg"])
        writer.writerows(data)

    print(f"Wrote {samples} samples at fs={fs} Hz to {filename}")
    return filename


def apply_kalman_to_commands(in_file, out_file, q=0.01, r=1.0):
    """
    Load commands, apply Kalman filter on azimuth and elevation,
    and write filtered outputs.
    """
    df = pd.read_csv(in_file)

    t = df['time'].values
    az = df['az_deg'].values
    el = df['el_deg'].values

    az_filt, az_rate = kalman_1d_angle(t, az, q=q, r=r)
    el_filt, el_rate = kalman_1d_angle(t, el, q=q, r=r)

    df['az_deg_filt'] = az_filt
    df['el_deg_filt'] = el_filt
    df['az_rate_filt'] = az_rate
    df['el_rate_filt'] = el_rate

    df.to_csv(out_file, index=False)
    print(f"Filtered commands written to {out_file} (q={q}, r={r})")
    return out_file


def generate_and_filter_commands(fs, samples, q, r,
                                 raw_file="sample_commands.csv",
                                 filt_file="sample_commands_filtered.csv"):
    """
    Convenience wrapper: generate trajectory and filter it.
    """
    raw = generate_samples_commands(fs, samples, filename=raw_file)
    filt = apply_kalman_to_commands(raw, filt_file, q=q, r=r)
    return filt


# ============================================================
# SIMULATION + METRICS
# ============================================================

def get_command(t_now, time_cmd, cmd_values):
    """
    Interpolate command value at time t_now.
    """
    return np.interp(t_now, time_cmd, cmd_values)


def simulate(
    kp,
    ki,
    kd,
    dt=0.05,
    sim_time=10.0,
    cmd_file='sample_commands_filtered.csv',
    tau=tau_default,
    servo_voltage=7.4,
    plot=False,
    return_logs=False,
):
    """
    Run full discrete-time simulation for pan/tilt servos following
    filtered command trajectories from CSV.
    """
    cmd_df = pd.read_csv(cmd_file)
    time_cmd = cmd_df['time'].values
    az_cmd = cmd_df['az_deg_filt'].values
    el_cmd = cmd_df['el_deg_filt'].values

    init_az = float(az_cmd[0])
    init_el = float(el_cmd[0])

    servo_pan = FirstOrderServo(init_angle=init_az, tau=tau, voltage=servo_voltage)
    servo_tilt = FirstOrderServo(init_angle=init_el, tau=tau, voltage=servo_voltage)

    prev_err_az, prev_err_el = 0.0, 0.0
    integral_az, integral_el = 0.0, 0.0

    time_log = []
    az_log, el_log = [], []
    az_err_log, el_err_log = [], []
    sp_az_log, sp_el_log = [], []

    t_vals = np.arange(0.0, sim_time, dt)
    for t in t_vals:
        curr_az = servo_pan.angle
        curr_el = servo_tilt.angle

        sp_az = get_command(t, time_cmd, az_cmd)
        sp_el = get_command(t, time_cmd, el_cmd)

        az_control, az_err, integral_az = pid_controller(
            sp_az, curr_az, kp, ki, kd, prev_err_az, integral_az, dt
        )
        el_control, el_err, integral_el = pid_controller(
            sp_el, curr_el, kp, ki, kd, prev_err_el, integral_el, dt
        )

        prev_err_az, prev_err_el = az_err, el_err

        az_cmd_state = np.clip(az_control, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        el_cmd_state = np.clip(el_control, ANGLE_MIN_DEG, ANGLE_MAX_DEG)

        pwm_az = angle_deg_to_pwm(az_cmd_state)
        pwm_el = angle_deg_to_pwm(el_cmd_state)

        new_az = servo_pan.step_pwm(pwm_az, dt)
        new_el = servo_tilt.step_pwm(pwm_el, dt)

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

    logs = {
        "time": time_log,
        "az": az_log,
        "el": el_log,
        "az_err": az_err_log,
        "el_err": el_err_log,
        "sp_az": sp_az_log,
        "sp_el": sp_el_log,
    }

    if plot:
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_log, az_log, label='Azimuth (current)')
        plt.plot(time_log, el_log, label='Elevation (current)')
        plt.plot(time_log, sp_az_log, '--', label='Azimuth (command)')
        plt.plot(time_log, sp_el_log, '--', label='Elevation (command)')
        plt.title('Servo Angles vs Time')
        plt.ylabel('Angle (deg)')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_log, az_err_log, label='Azimuth Error')
        plt.plot(time_log, el_err_log, label='Elevation Error')
        plt.title('Tracking Error vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("mags_sim_plot.png")
        print("Saved plot to mags_sim_plot.png")

    if return_logs:
        return logs
    return None


def compute_tracking_metrics(logs, big_err_deg=5.0):
    """
    Compute tracking metrics:
      - max az error
      - max el error
      - IAE total (az + el)
      - time with |err| > big_err_deg
    """
    t = logs["time"]
    az = logs["az"]
    el = logs["el"]
    sp_az = logs["sp_az"]
    sp_el = logs["sp_el"]

    e_az = sp_az - az
    e_el = sp_el - el
    dt = t[1] - t[0] if len(t) > 1 else 0.0

    max_err_az = np.max(np.abs(e_az))
    max_err_el = np.max(np.abs(e_el))

    iae = np.sum(np.abs(e_az) + np.abs(e_el)) * dt

    max_err = np.maximum(np.abs(e_az), np.abs(e_el))
    big_time = np.sum(max_err > big_err_deg) * dt

    return {
        "max_err_az": max_err_az,
        "max_err_el": max_err_el,
        "IAE_total": iae,
        "time_big_err": big_time,
    }


def print_metrics(label, metrics):
    print(f"\n[{label}]")
    print(f"  Max az error    : {metrics['max_err_az']:.2f} deg")
    print(f"  Max el error    : {metrics['max_err_el']:.2f} deg")
    print(f"  IAE (total)     : {metrics['IAE_total']:.2f} deg·s")
    print(f"  Time |err|>5°   : {metrics['time_big_err']:.2f} s")


# ============================================================
# EXPERIMENTS
# ============================================================

def experiment_dt_sweep(kp, ki, kd,
                        dt_list,
                        sim_time,
                        cmd_file,
                        tau=tau_default,
                        servo_voltage=7.4):
    """
    Sweep control-loop dt and plot tracking metrics vs dt.
    """
    results = []

    for dt in dt_list:
        logs = simulate(
            kp, ki, kd,
            dt=dt,
            sim_time=sim_time,
            cmd_file=cmd_file,
            tau=tau,
            servo_voltage=servo_voltage,
            plot=False,
            return_logs=True,
        )
        metrics = compute_tracking_metrics(logs)
        print_metrics(f"dt={dt:.3f}s", metrics)

        row = {"dt": dt}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)

    # IAE vs dt
    plt.figure()
    plt.plot(df["dt"], df["IAE_total"], marker='o')
    plt.xlabel("Control-loop dt (s)")
    plt.ylabel("IAE_total (deg·s)")
    plt.title("IAE vs Control-loop dt")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_dt_IAE.png")
    print("Saved IAE vs dt plot to exp_dt_IAE.png")

    # Time |err|>5° vs dt
    plt.figure()
    plt.plot(df["dt"], df["time_big_err"], marker='o')
    plt.xlabel("Control-loop dt (s)")
    plt.ylabel("Time |err|>5° (s)")
    plt.title("Big-error time vs dt")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_dt_bigerr.png")
    print("Saved big-error vs dt plot to exp_dt_bigerr.png")

    return df


def experiment_servo_voltage_sweep(kp, ki, kd,
                                   voltage_list,
                                   dt,
                                   sim_time,
                                   cmd_file,
                                   tau=tau_default):
    """
    Sweep servo supply voltage and see impact on tracking.
    """
    results = []

    for V in voltage_list:
        logs = simulate(
            kp, ki, kd,
            dt=dt,
            sim_time=sim_time,
            cmd_file=cmd_file,
            tau=tau,
            servo_voltage=V,
            plot=False,
            return_logs=True,
        )
        metrics = compute_tracking_metrics(logs)
        print_metrics(f"servo_voltage={V:.2f}V", metrics)

        row = {"servo_voltage": V}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)

    plt.figure()
    plt.plot(df["servo_voltage"], df["IAE_total"], marker='o')
    plt.xlabel("Servo voltage (V)")
    plt.ylabel("IAE_total (deg·s)")
    plt.title("IAE vs Servo Voltage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_voltage_IAE.png")
    print("Saved IAE vs voltage plot to exp_voltage_IAE.png")

    return df


def experiment_fs_sweep(kp, ki, kd,
                        fs_list,
                        samples,
                        q,
                        r,
                        dt,
                        sim_time,
                        raw_file="sample_commands.csv",
                        filt_file="sample_commands_filtered.csv",
                        tau=tau_default,
                        servo_voltage=7.4):
    """
    Sweep the command sampling rate fs and see effect on tracking.
    """
    results = []

    for fs in fs_list:
        filt_cmd_file = generate_and_filter_commands(
            fs, samples, q=q, r=r,
            raw_file=raw_file,
            filt_file=filt_file
        )

        logs = simulate(
            kp, ki, kd,
            dt=dt,
            sim_time=sim_time,
            cmd_file=filt_cmd_file,
            tau=tau,
            servo_voltage=servo_voltage,
            plot=False,
            return_logs=True,
        )
        metrics = compute_tracking_metrics(logs)
        print_metrics(f"fs={fs:.1f}Hz", metrics)

        row = {"fs": fs}
        row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)

    plt.figure()
    plt.plot(df["fs"], df["IAE_total"], marker='o')
    plt.xlabel("Command sampling fs (Hz)")
    plt.ylabel("IAE_total (deg·s)")
    plt.title("IAE vs Command Sampling Rate fs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("exp_fs_IAE.png")
    print("Saved IAE vs fs plot to exp_fs_IAE.png")

    return df


def experiment_kalman_sweep(kp, ki, kd,
                            q_list,
                            r_list,
                            fs,
                            samples,
                            dt,
                            sim_time,
                            raw_file="sample_commands.csv",
                            filt_file="sample_commands_filtered.csv",
                            tau=tau_default,
                            servo_voltage=7.4):
    """
    Sweep Kalman (q, r) and plot metrics over the grid.
    """
    results = []

    # Generate raw commands once
    raw_cmd_file = generate_samples_commands(fs, samples, filename=raw_file)

    for q in q_list:
        for r in r_list:
            filt_cmd_file = apply_kalman_to_commands(
                in_file=raw_cmd_file,
                out_file=filt_file,
                q=q,
                r=r,
            )

            logs = simulate(
                kp, ki, kd,
                dt=dt,
                sim_time=sim_time,
                cmd_file=filt_cmd_file,
                tau=tau,
                servo_voltage=servo_voltage,
                plot=False,
                return_logs=True,
            )
            metrics = compute_tracking_metrics(logs)
            label = f"q={q}, r={r}"
            print_metrics(label, metrics)

            row = {"q": q, "r": r}
            row.update(metrics)
            results.append(row)

    df = pd.DataFrame(results)

    # Example: heatmap-like scatter of IAE vs q,r
    plt.figure()
    scatter = plt.scatter(df["q"], df["r"], c=df["IAE_total"])
    plt.xlabel("q (process noise)")
    plt.ylabel("r (measurement noise)")
    plt.title("IAE_total for Kalman (q,r)")
    plt.colorbar(scatter, label="IAE_total (deg·s)")
    plt.tight_layout()
    plt.savefig("exp_kalman_qr_IAE.png")
    print("Saved Kalman (q,r) IAE plot to exp_kalman_qr_IAE.png")

    return df


## ============================================================
# MAIN: 
# ============================================================

if __name__ == "__main__":
    # Baseline settings (starting guesses)
    base_fs = 25.0      # command sampling rate (Hz)
    base_samples = 500  # number of command samples
    base_q = 0.01
    base_r = 1.0
    base_dt = 0.05
    base_sim_time = 10.0
    base_voltage = 7.4

    # --------------------------------------------------------
    # generate commands + Kalman with baseline params
    # --------------------------------------------------------
    cmd_file = generate_and_filter_commands(
        fs=base_fs,
        samples=base_samples,
        q=base_q,
        r=base_r,
        raw_file="sample_commands.csv",
        filt_file="sample_commands_filtered.csv",
    )

    print("\nTuning PID on continuous first-order servo model...")
    best_kp, best_ki, best_kd = tune_pid_step()

    print("\nRunning baseline simulation...")
    baseline_logs = simulate(
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
        dt=base_dt,
        sim_time=base_sim_time,
        cmd_file=cmd_file,
        tau=tau_default,
        servo_voltage=base_voltage,
        plot=False,
        return_logs=True,
    )
    baseline_metrics = compute_tracking_metrics(baseline_logs)
    print_metrics("Baseline", baseline_metrics)

    # --------------------------------------------------------
    # run experiments and collect DataFrames
    # --------------------------------------------------------

    # --- Experiment: servo voltage sweep ---
    voltage_list = [6.0, 6.7, 7.4]
    df_voltage = experiment_servo_voltage_sweep(
        best_kp, best_ki, best_kd,
        voltage_list=voltage_list,
        dt=base_dt,
        sim_time=base_sim_time,
        cmd_file=cmd_file,
        tau=tau_default,
    )

    # --- Experiment: command sampling rate fs sweep ---
    fs_list = [20,22.5,25,27.5,30,32.5, 35, 37.5,40]
    df_fs = experiment_fs_sweep(
        best_kp, best_ki, best_kd,
        fs_list=fs_list,
        samples=base_samples,
        q=base_q,
        r=base_r,
        dt=base_dt,
        sim_time=base_sim_time,
        raw_file="sample_commands.csv",
        filt_file="sample_commands_filtered.csv",
        tau=tau_default,
        servo_voltage=base_voltage,
    )

    # --- Experiment: Kalman (q,r) sweep ---
    q_list = [1e-5, 3e-5, 1e-4, 3e-4,
          1e-3, 3e-3, 1e-2, 3e-2,
          1e-1]

    r_list = [0.01, 0.03, 0.1, 0.3,
          1.0, 3.0, 5.0, 10.0]
    df_kalman = experiment_kalman_sweep(
        best_kp, best_ki, best_kd,
        q_list=q_list,
        r_list=r_list,
        fs=base_fs,
        samples=base_samples,
        dt=base_dt,
        sim_time=base_sim_time,
        raw_file="sample_commands.csv",
        filt_file="sample_commands_filtered.csv",
        tau=tau_default,
        servo_voltage=base_voltage,
    )

    # --------------------------------------------------------
    #  extract BEST parameters from these experiments
    # --------------------------------------------------------

    # Best servo voltage
    idx_v = df_voltage["IAE_total"].idxmin()
    best_voltage = float(df_voltage.loc[idx_v, "servo_voltage"])

    # Best command sampling fs
    idx_fs = df_fs["IAE_total"].idxmin()
    best_fs = float(df_fs.loc[idx_fs, "fs"])

    # Best Kalman q,r
    idx_k = df_kalman["IAE_total"].idxmin()
    best_q = float(df_kalman.loc[idx_k, "q"])
    best_r = float(df_kalman.loc[idx_k, "r"])

    print("\n===== BEST PARAMETERS (voltage, fs, q, r) =====")
    print(f"  best_fs       = {best_fs:.2f} Hz")
    print(f"  best_voltage  = {best_voltage:.2f} V")
    print(f"  best_q        = {best_q}")
    print(f"  best_r        = {best_r}")
    print("================================================")

    # --------------------------------------------------------
    # regenerate commands with best fs, q, r
    # --------------------------------------------------------
    best_cmd_file = generate_and_filter_commands(
        fs=best_fs,
        samples=base_samples,
        q=best_q,
        r=best_r,
        raw_file="sample_commands_best.csv",
        filt_file="sample_commands_filtered_best.csv",
    )

    # --------------------------------------------------------
    #  dt sweep on this "best-commands" configuration
    # --------------------------------------------------------
    dt_list = [0.111, 0.112,0.113,0.114,0.115,0.116, 0.117,0.118,0.119,0.12,0.121,0.122, 0.123,0.124, 0.125]
    df_dt = experiment_dt_sweep(
        best_kp, best_ki, best_kd,
        dt_list=dt_list,
        sim_time=base_sim_time,
        cmd_file=best_cmd_file,
        tau=tau_default,
        servo_voltage=best_voltage,
    )

    idx_dt = df_dt["IAE_total"].idxmin()
    best_dt = float(df_dt.loc[idx_dt, "dt"])

    print("\n===== BEST dt FROM SWEEP =====")
    print(f"  best_dt       = {best_dt:.3f} s")
    print("================================")

    # --------------------------------------------------------
    # Final suim with  optimal params
    # --------------------------------------------------------
    print("\nRunning FINAL MAGS simulation with optimal parameters...")
    final_logs = simulate(
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
        dt=best_dt,
        sim_time=base_sim_time,
        cmd_file=best_cmd_file,
        tau=tau_default,
        servo_voltage=best_voltage,
        plot=True,
        return_logs=True,
    )
    final_metrics = compute_tracking_metrics(final_logs)
    print_metrics("Final (optimal)", final_metrics)

    print("\nAll experiments and final optimal simulation complete.")
