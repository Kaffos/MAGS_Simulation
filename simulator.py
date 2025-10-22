"""Simple antenna tracking + RF link simulator

Reads the two provided CSVs and runs a basic simulation for the antenna mount
tracking the rocket. Produces time-series plots for elevation, tracking error,
rocket path and received power / SNR.

This is intentionally simple and easy to extend (PID tuning, sensor models,
filters, 3D geometry, real antenna pattern, modulation models, etc.).
"""
import math
import collections
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp


FT_TO_M = 0.3048


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, integrator_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrator = 0.0
        self.prev_error = None
        self.integrator_limit = integrator_limit

    def reset(self):
        self.integrator = 0.0
        self.prev_error = None

    def step(self, error, dt):
        # Proportional
        p = self.kp * error
        # Integral
        self.integrator += error * dt
        if self.integrator_limit is not None:
            self.integrator = max(min(self.integrator, self.integrator_limit), -self.integrator_limit)
        i = self.ki * self.integrator
        # Derivative
        d = 0.0
        if self.prev_error is not None and dt > 0:
            d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        return p + i + d


class AntennaMount:
    def __init__(self, angle0_rad=0.0, max_speed_rad_s=math.radians(30), max_accel_rad_s2=math.radians(100)):
        self.angle = float(angle0_rad)
        self.ang_vel = 0.0
        self.max_speed = float(max_speed_rad_s)
        self.max_accel = float(max_accel_rad_s2)

    def apply_command(self, target_rate, dt):
        """Apply commanded angular rate (rad/s) with acceleration and speed limits.

        target_rate: desired angular velocity (rad/s)
        dt: timestep seconds
        """
        # limit acceleration
        rate_change = target_rate - self.ang_vel
        max_drate = self.max_accel * dt
        rate_change = max(-max_drate, min(max_drate, rate_change))
        self.ang_vel += rate_change
        # limit speed
        self.ang_vel = max(-self.max_speed, min(self.max_speed, self.ang_vel))
        # integrate
        self.angle += self.ang_vel * dt
        # normalise angle to -pi..pi for numerical stability (optional)
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi


def fspl_db(distance_m, freq_hz):
    """Free-space path loss (dB)."""
    if distance_m <= 0:
        return 0.0
    c = 299792458.0
    lam = c / freq_hz
    return 20 * math.log10(4 * math.pi * distance_m / lam)


def simple_yagi_gain_db(off_axis_rad, g_max_dbi=9.0):
    """Very simple Yagi pattern: max gain at boresight, roll-off with off-axis.

    This is an approximation: use cos^n pattern for main lobe.
    """
    # keep gain positive; reduce according to cos^6 pattern to make main lobe narrow
    loss_lin = max(0.0, math.cos(off_axis_rad)) ** 6
    g_lin_max = 10 ** (g_max_dbi / 10.0)
    g_lin = max(1e-3, g_lin_max * loss_lin)
    return 10 * math.log10(g_lin)


def run_simulation(
    dist_to_pad_ft=900,
    use_10k=True,
    pid_params=(2.0, 0.0, 0.1),
    antenna_max_speed_dps=30.0,
    antenna_max_accel_dps2=100.0,
    tx_power_dbm=14.0,
    freq_hz=915e6,
    rx_antenna_gain_dbi=9.0,
    tx_antenna_gain_dbi=0.0,
    noise_nf_db=6.0,
    bw_hz=125e3,
):
    """Run the simulation and return a DataFrame with logged states.

    Uses the 10k Porthos flight as primary timebase; loads CSVs from the script directory.
    """
    scripts_dir = osp.dirname(__file__)
    data_dir = osp.join(scripts_dir, 'flight_data')
    df10 = pd.read_csv(data_dir + '\\MRT Porthos official Flight data 2023.csv')
    df60 = pd.read_csv(data_dir + '\\FlightData60k.csv')


    # use 10k as timebase unless requested otherwise
    if use_10k:
        t = df10['Time'].astype(float).to_numpy()
        alt = df10['Altitude'].astype(float).to_numpy()
        vel = df10['Velocity'].astype(float).to_numpy()
        label10 = '10k'
        df_other = df60
    else:
        t = df60['Time'].astype(float).to_numpy()
        alt = df60['Altitude'].astype(float).to_numpy()
        # guess velocity column name
        if 'Total Velocity' in df60.columns:
            vel = df60['Total Velocity'].astype(float).to_numpy()
        else:
            vel = np.gradient(alt, t)
        label10 = '60k'
        df_other = df10

    # convert pad distance to meters
    dist_to_pad_m = dist_to_pad_ft * FT_TO_M

    # desired elevation angle
    desired_theta = np.arctan2(alt, dist_to_pad_m)

    # instantiate antenna
    mount = AntennaMount(angle0_rad=0.0, max_speed_rad_s=math.radians(antenna_max_speed_dps), max_accel_rad_s2=math.radians(antenna_max_accel_dps2))

    pid = PIDController(*pid_params, integrator_limit=math.radians(30))

    # logging
    rows = []

    # RF constants
    k_b = 1.380649e-23
    noise_floor_dbm = -174.0 + 10 * math.log10(bw_hz) + noise_nf_db

    # simulation loop
    for i in range(len(t)):
        if i == 0:
            dt = 0.0
        else:
            dt = t[i] - t[i - 1]
        if dt <= 0:
            dt = 1e-6

        targ = desired_theta[i]
        error = (targ - mount.angle)
        # wrap error to -pi..pi
        error = (error + math.pi) % (2 * math.pi) - math.pi

        # PID computes desired angular rate
        commanded_rate = pid.step(error, dt)

        # apply to mount
        mount.apply_command(commanded_rate, dt)

        # tracking error
        track_error = targ - mount.angle
        track_error = (track_error + math.pi) % (2 * math.pi) - math.pi

        # compute RF link
        # 3D distance from Tx(rocket) to Rx(antenna) approximated: antenna at ground at horizontal dist_to_pad_m
        dist3 = math.hypot(dist_to_pad_m, alt[i])

        # antenna pointing error = difference between boresight (mount.angle) and ideal elev angle
        off_axis = abs(track_error)

        rx_gain_db = simple_yagi_gain_db(off_axis, g_max_dbi=rx_antenna_gain_dbi)
        fspl = fspl_db(dist3, freq_hz)

        pr_dbm = tx_power_dbm + tx_antenna_gain_dbi + rx_gain_db - fspl
        snr_db = pr_dbm - noise_floor_dbm

        rows.append({
            'time': t[i],
            'altitude_m': alt[i],
            'desired_theta_rad': targ,
            'desired_theta_deg': math.degrees(targ),
            'mount_theta_rad': mount.angle,
            'mount_theta_deg': math.degrees(mount.angle),
            'mount_ang_vel_rad_s': mount.ang_vel,
            'mount_ang_vel_deg_s': math.degrees(mount.ang_vel),
            'tracking_error_deg': math.degrees(track_error),
            'distance_m': dist3,
            'rx_gain_dbi': rx_gain_db,
            'fspl_db': fspl,
            'pr_dbm': pr_dbm,
            'snr_db': snr_db,
        })

    logdf = pd.DataFrame(rows)
    return logdf


def plot_results(logdf, show=True, out_prefix='sim'):
    t = logdf['time']

    plt.figure(figsize=(10, 5))
    plt.plot(t, logdf['desired_theta_deg'], label='Desired elevation (deg)')
    plt.plot(t, logdf['mount_theta_deg'], label='Mount elevation (deg)')
    plt.xlabel('Time (s)')
    plt.ylabel('Elevation (deg)')
    plt.title('Elevation tracking')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_elevation.png")

    plt.figure(figsize=(10, 4))
    plt.plot(t, logdf['tracking_error_deg'])
    plt.xlabel('Time (s)')
    plt.ylabel('Tracking error (deg)')
    plt.title('Tracking error (deg)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_tracking_error.png")

    plt.figure(figsize=(8, 6))
    plt.plot(logdf['distance_m'], logdf['desired_theta_deg'], label='Rocket elevation (deg)')
    plt.plot([0], [0], '.')
    plt.xlabel('Horizontal distance (m)')
    plt.ylabel('Elevation (deg)')
    plt.title('Rocket elevation vs horizontal distance (sample)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rocket_path.png")

    plt.figure(figsize=(10, 4))
    plt.plot(t, logdf['pr_dbm'], label='Received power (dBm)')
    plt.plot(t, logdf['snr_db'], label='SNR (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('dB / dBm')
    plt.title('RF link metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_rf.png")

    if show:
        plt.show()


if __name__ == '__main__':
    print('Running simulator...')
    log = run_simulation()
    print('Sim finished. Saving logs to sim_log.csv')
    log.to_csv('sim_log.csv', index=False)
    plot_results(log)
