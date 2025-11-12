import numpy as np
import matplotlib.pyplot as plt

# ---------------- PID Controller Block ----------------
def pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt):
    error = setpoint - pv
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, error, integral

# ---------------- Mappings  ----------------
def pwm_to_angle(pwm, pwm_min=800, pwm_max=2200, angle_min=-90, angle_max=90):
    angle = angle_min + ((pwm - pwm_min) / (pwm_max - pwm_min)) * (angle_max - angle_min)
    return np.clip(angle, angle_min, angle_max)

def angle_to_pwm(angle, pwm_min=800, pwm_max=2200, angle_min=-90, angle_max=90):
    pwm = pwm_min + ((angle - angle_min) / (angle_max - angle_min)) * (pwm_max - pwm_min)
    return np.clip(pwm, pwm_min, pwm_max)

# ---------------- First-order Servo Plant ----------------

#servo model is very subject to change once we have legitimate hardware
class FirstOrderServo:

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

# ---------------- Simulation ----------------
def simulate(kp=0.8, ki=0.2, kd=0.05, dt=0.05, sim_time=15, az_cmd=45, el_cmd=45,
             angle_limits=(-90, 90), dcmd_limit_per_step=5.0):
    
    # Initialize servo simulation
    servo_pan = FirstOrderServo(init_angle=0)
    servo_tilt = FirstOrderServo(init_angle=0)

    # Initialize PID controllers
    prev_err_az, prev_err_el = 0.0, 0.0
    integral_az, integral_el = 0.0, 0.0

    # Logs
    time_log = []
    az_log, el_log = [], []
    az_err_log, el_err_log = [], []

    #  start state at current angle
    az_cmd_state = servo_pan.angle
    el_cmd_state = servo_tilt.angle

    for t in np.arange(0, sim_time, dt):
        curr_az = servo_pan.angle
        curr_el = servo_tilt.angle

        # Azimuth motor PID initiation
        az_control, az_err, integral_az = pid_controller(az_cmd, curr_az, kp, ki, kd, prev_err_az, integral_az, dt)
        # Elevation motor PID initiation
        el_control, el_err, integral_el = pid_controller(el_cmd, curr_el, kp, ki, kd, prev_err_el, integral_el, dt)
        
        # update our error vals
        prev_err_az, prev_err_el = az_err, el_err

        # PID output is a small increment
        az_delta = np.clip(az_control, -dcmd_limit_per_step, dcmd_limit_per_step)
        el_delta = np.clip(el_control, -dcmd_limit_per_step, dcmd_limit_per_step)

        # Update commanded angles
        az_cmd_state = np.clip(az_cmd_state + az_delta, angle_limits[0], angle_limits[1])
        el_cmd_state = np.clip(el_cmd_state + el_delta, angle_limits[0], angle_limits[1])

        # Update toward commanded angles
        new_az = servo_pan.step(az_cmd_state, dt)
        new_el = servo_tilt.step(el_cmd_state, dt)

        # Log for plot
        time_log.append(t)
        az_log.append(new_az)
        el_log.append(new_el)
        az_err_log.append(az_err)
        el_err_log.append(el_err)

    # ---------- PLOTS ----------
    plt.figure(figsize=(10, 6))

    # Elevation and Azimuth Angles Convergence over time
    plt.subplot(2, 1, 1)
    plt.plot(time_log, az_log, label='Current Azimuth (°)')
    plt.axhline(az_cmd, linestyle='--', label='Commanded Azimuth')
    plt.plot(time_log, el_log, label='Current Elevation (°)')
    plt.axhline(el_cmd, linestyle='--', label='Commanded Elevation')
    plt.title('Servo Angles vs Time')
    plt.ylabel('Angle (°)')
    plt.legend()
    plt.grid(True)

    # Error Plot over time
    plt.subplot(2, 1, 2)
    plt.plot(time_log, az_err_log, label='Azimuth Error (°)')
    plt.plot(time_log, el_err_log, label='Elevation Error (°)')
    plt.title('Tracking Error vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (°)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Sample Simulation
simulate(kp=0.8, ki=0.02, kd=0.05, dt=0.05, sim_time=10, az_cmd=50, el_cmd=15)
