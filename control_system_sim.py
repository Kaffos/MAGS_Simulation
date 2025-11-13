import numpy as np
import matplotlib.pyplot as plt
import control as ct

# =====================================================
# 1) Continuous-time model using python-control
#    (for design/tuning)
# =====================================================

# ----- Servo model (first-order) -----
# G_servo(s) = 1 / (tau s + 1)
tau = 0.15  # seconds (same idea as in your FirstOrderServo model)
G_servo = ct.tf([1], [tau, 1])  # 1 / (tau s + 1)


def make_pid_tf(kp, ki, kd):
    """
    PID controller in transfer function form.
    C(s) = Kp + Ki/s + Kd*s
         = (Kd*s^2 + Kp*s + Ki) / s
    """
    num = [kd, kp, ki]
    den = [1, 0]   # 's'
    return ct.tf(num, den)


def make_closed_loop(kp, ki, kd):
    """
    Closed-loop transfer function from reference to servo angle.
    T(s) = feedback(C(s)*G(s), 1)
    """
    C = make_pid_tf(kp, ki, kd)
    L = C * G_servo          # open-loop
    T = ct.feedback(L, 1)    # unity feedback
    return T


def settling_time_from_response(t, y, tol=0.02):
    """
    Rough settling time for a step of magnitude 1.
    First time after which |y-1| < tol forever.
    """
    y_err = np.abs(y - 1.0)
    for i in range(len(t)):
        if np.all(y_err[i:] < tol):
            return t[i]
    return t[-1]


def tune_pid():
    """
    Very simple grid search over PID gains using python-control
    on the first-order servo model.
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

    print("\nBest PID (continuous model):", best_pid)
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


# =====================================================
# 2) Discrete-time MAGS-style simulation
#    (your FirstOrderServo + PID controller)
# =====================================================

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
    Simple first-order servo model in discrete time.
    angle[k+1] = angle[k] + rate * dt
    where rate ~ (cmd - angle) / tau, limited by rate_limit.
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


def get_commands_from_schedule(t, default_az, default_el, cmd_schedule=None):
    """
    cmd_schedule: optional list of (time, az_cmd, el_cmd)
    Returns the commanded (az, el) at time t.
    If no schedule is provided, uses default_az / default_el.
    """
    if cmd_schedule is None or len(cmd_schedule) == 0:
        return default_az, default_el

    az, el = default_az, default_el
    for t_i, az_i, el_i in cmd_schedule:
        if t >= t_i:
            az, el = az_i, el_i
        else:
            break
    return az, el


def simulate_discrete(kp=0.8, ki=0.02, kd=0.05,
                      dt=0.05, sim_time=10.0,
                      az_cmd=50.0, el_cmd=15.0,
                      angle_limits=(-90, 90),
                      dcmd_limit_per_step=5.0,
                      cmd_schedule=None,
                      plot=True,
                      return_logs=False):
    """
    Discrete-time 2D (az, el) servo system with PID control.
    """

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
    sp_az_log, sp_el_log = [], []   # NEW: commanded angles log

    # Commanded states (servo setpoints)
    az_cmd_state = servo_pan.angle
    el_cmd_state = servo_tilt.angle

    t_vals = np.arange(0.0, sim_time, dt)
    for t in t_vals:
        curr_az = servo_pan.angle
        curr_el = servo_tilt.angle

        # Get setpoints (constant or from schedule)
        sp_az, sp_el = get_commands_from_schedule(
            t, az_cmd, el_cmd, cmd_schedule
        )

        # PID control
        az_control, az_err, integral_az = pid_controller(
            sp_az, curr_az, kp, ki, kd, prev_err_az, integral_az, dt
        )
        el_control, el_err, integral_el = pid_controller(
            sp_el, curr_el, kp, ki, kd, prev_err_el, integral_el, dt
        )

        # Update error memory
        prev_err_az, prev_err_el = az_err, el_err

        # Limit increments in commanded angle
        az_delta = np.clip(az_control, -dcmd_limit_per_step, dcmd_limit_per_step)
        el_delta = np.clip(el_control, -dcmd_limit_per_step, dcmd_limit_per_step)

        # Update commanded angles (within limits)
        az_cmd_state = np.clip(az_cmd_state + az_delta,
                               angle_limits[0], angle_limits[1])
        el_cmd_state = np.clip(el_cmd_state + el_delta,
                               angle_limits[0], angle_limits[1])

        # Servo physical step
        new_az = servo_pan.step(az_cmd_state, dt)
        new_el = servo_tilt.step(el_cmd_state, dt)

        # Log
        time_log.append(t)
        az_log.append(new_az)
        el_log.append(new_el)
        az_err_log.append(az_err)
        el_err_log.append(el_err)
        sp_az_log.append(sp_az)   # NEW
        sp_el_log.append(sp_el)   # NEW

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

        # Plot commanded signals (works for both constant & schedule)
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
        plt.show()

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

# =====================================================
# 3) Main: design with control, then test in discrete sim
# =====================================================

if __name__ == "__main__":
    # --- A) Tune PID on the continuous-time model using python-control ---
    print("Tuning PID on continuous first-order servo model...")
    best_kp, best_ki, best_kd = tune_pid()

    # Plot continuous-time step response with best gains
    #plot_continuous_step_response(best_kp, best_ki, best_kd)

    # --- B) Use those same gains in the discrete MAGS-style simulation ---
    #print("\nRunning discrete-time MAGS-style simulation with tuned PID...")
    #simulate_discrete(
      #  kp=best_kp,
       # ki=best_ki,
        #kd=best_kd,
        #dt=0.05,
       # sim_time=10.0,
       # az_cmd=50.0,
       # el_cmd=15.0,
       # plot=True,
       # return_logs=False,
  #  )

    # --- C) Example of time-varying command schedule (uncomment to try) ---
 
    cmd_schedule = [
        (0.0,  50.0,  15.0),   # from t = 0s, command (50, 15)
        (5.0, -20.0,  10.0),   # from t = 5s, command (-20, 10)
        (8.0,  60.0,   0.0),   # from t = 8s, command (60, 0)
    ]

    simulate_discrete(
        kp=best_kp,
        ki=best_ki,
        kd=best_kd,
        dt=0.05,
        sim_time=12.0,
        az_cmd=0.0,
        el_cmd=0.0,
        cmd_schedule=cmd_schedule,
        plot=True,
        return_logs=False,
    )

