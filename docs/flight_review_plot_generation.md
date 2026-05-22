# PX4 Flight Review Plot Generation

This document summarizes how `ref/flight_review` generates plots, where plot
data comes from, and what transformations are applied before rendering. It is
based on these reference files:

- `ref/flight_review/app/plot_app/main.py`
- `ref/flight_review/app/plot_app/helper.py`
- `ref/flight_review/app/plot_app/configured_plots.py`
- `ref/flight_review/app/plot_app/plotting.py`
- `ref/flight_review/app/plot_app/downsampling.py`
- `ref/flight_review/app/plot_app/pid_analysis_plots.py`
- `ref/flight_review/app/plot_app/pid_analysis.py`
- `ref/flight_review/app/plot_app/statistics_plots.py`
- `ref/flight_review/app/plot_app/overview_generator.py`

## High-Level Flow

Flight Review is a Bokeh server application. A normal log plot page follows this
flow:

1. `plot_app/main.py` receives a request for `/plot_app?log=<id>`.
2. It resolves the log id to a `.ulg` file with `get_log_filename()`.
3. It loads the log using `load_ulog_file()` from `helper.py`.
4. It wraps the ULog with `PX4ULog`, then calls `px4_ulog.add_roll_pitch_yaw()`.
5. It reads optional DB metadata from `Logs` and `Vehicle`.
6. It calls `configured_plots.generate_plots(...)`.
7. `generate_plots()` builds Bokeh plot objects using `DataPlot`, `DataPlot2D`,
   `DataPlotSpec`, `DataPlotFFT`, and `plot_map()`.
8. The Bokeh objects are attached to the current document and rendered through
   the HTML templates.

The loaded `ULog` object is cached in RAM by `load_ulog_file()` using
`functools.lru_cache(maxsize=get_log_cache_size())`. This matters because ULog
loading is expensive and Bokeh reloads `main.py` per session.

## ULog Topics Loaded

`helper.load_ulog_file()` does not load every topic. It passes a `msg_filter` to
`pyulog.ULog`, so only selected topics are available to the default plot page.
The loaded topic list includes:

- `battery_status`
- `distance_sensor`
- `esc_status`
- `estimator_status`
- `sensor_combined`
- `cpuload`
- `vehicle_gps_position`
- `vehicle_local_position`
- `vehicle_local_position_setpoint`
- `vehicle_global_position`
- `actuator_controls_0`
- `actuator_controls_1`
- `actuator_outputs`
- `vehicle_angular_velocity`
- `vehicle_attitude`
- `vehicle_attitude_setpoint`
- `vehicle_rates_setpoint`
- `rc_channels`
- `position_setpoint_triplet`
- `vehicle_attitude_groundtruth`
- `vehicle_local_position_groundtruth`
- `vehicle_visual_odometry`
- `vehicle_status`
- `airspeed`
- `airspeed_validated`
- `manual_control_setpoint`
- `rate_ctrl_status`
- `vehicle_air_data`
- `vehicle_magnetometer`
- `system_power`
- `tecs_status`
- `sensor_baro`
- `sensor_accel`
- `sensor_accel_fifo`
- `sensor_gyro_fifo`
- `vehicle_angular_acceleration`
- `ekf2_timestamps`
- `manual_control_switches`
- `event`
- `vehicle_imu_status`
- `actuator_motors`
- `actuator_servos`
- `vehicle_thrust_setpoint`
- `vehicle_torque_setpoint`
- `failsafe_flags`

If a configured plot needs a topic or field that is absent, the plot wrapper
marks the plot as failed and `finalize()` returns `None`, so that plot is simply
not shown.

## Core Plot Wrappers

### `DataPlot`

`DataPlot` is the standard time-series plot wrapper. It:

- Selects a ULog dataset by `topic name` and `multi_id`.
- Uses the dataset `timestamp` field as the x-axis.
- Adds one or more y-axis series with `add_graph()`.
- Accepts either direct field names such as `vehicle_attitude.roll`, or lambda
  transformations that return `(derived_name, derived_array)`.
- Optionally marks NaN spans.
- Optionally overlays changed-parameter labels.
- Uses `DynamicDownsample` for server-side time-series downsampling.
- Formats x-axis timestamps as flight time in minutes/seconds.

### `DataPlot2D`

`DataPlot2D` renders x/y traces instead of time-series data. The main use is the
local-position plot shown above the Leaflet map:

- x-axis and y-axis are selected fields from one dataset.
- NaNs are removed.
- The first successful line can force equal aspect ratio. For the local-position
  plot, Flight Review uses the `Estimated` trace to set the initial x/y ranges
  with `plot_set_equal_aspect_ratio()`, a minimum span of 5 meters, and a 1.3
  zoom-out factor.
- The local-position plot calls `add_graph('y', 'x', ...)`, so the horizontal
  axis is local `y` and the vertical axis is local `x`.
- It does not use dynamic downsampling.

### `DataPlotSpec`

`DataPlotSpec` renders a spectrogram-like power spectral density image. It:

- Uses `timestamp_sample` if present, otherwise `timestamp`.
- Estimates sampling frequency from first and last timestamps.
- Requires sampling frequency >= 100 Hz.
- Uses `scipy.signal.spectrogram()`.
- Sums PSD across all requested axes.
- Converts power to dB with `10 * log10(sum_psd)`.
- Downsamples the image horizontally if there are too many time bins.

### `DataPlotFFT`

`DataPlotFFT` renders FFT amplitude plots. It:

- Uses `timestamp_sample` if present, otherwise `timestamp`.
- Requires sampling frequency >= 100 Hz.
- Computes FFT with `pyfftw.interfaces.numpy_fft.fft()`.
- Uses `scipy.fftpack.fftfreq()` for frequency bins.
- Plots only the positive-frequency half.
- Adds a mean-amplitude line above 40 Hz.
- Can mark filter/notch parameter frequencies such as `IMU_GYRO_CUTOFF`.

### `plot_map()`

`plot_map()` renders GPS path data from `vehicle_gps_position`.

For plain maps:

- Reads lat/lon from `vehicle_gps_position`.
- Filters to `fix_type > 2`.
- Uses `get_lat_lon_alt_deg()`:
  - New format: `latitude_deg`, `longitude_deg`, `altitude_msl_m`
  - Old format: `lat / 1e7`, `lon / 1e7`, `alt / 1e3`
- Projects WGS84 lat/lon into local meters using `map_projection()`.
- Uses `vehicle_local_position.ref_lat`, `ref_lon`, and `ref_timestamp` as the
  projection anchor when available.
- Optionally overlays `position_setpoint_triplet.current.lat`,
  `current.lon`, and `current.alt`.

## Shared Overlays

Most default time-series plots call `plot_flight_modes_background()`.

The flight-mode background comes from:

- Topic: `vehicle_status`
- Field: `nav_state`
- Extracted by `get_flight_mode_changes()`, which calls
  `ulog.get_dataset('vehicle_status').list_value_changes('nav_state')`
- The final marker is `(ulog.last_timestamp, -1)`.
- Colors come from `config_tables.flight_modes_table`; Flight Review renders
  the background boxes with `fill_alpha=0.09`.

For VTOL logs, an additional VTOL-mode band is added using:

- Topic: `vehicle_status`
- Fields:
  - `is_vtol`
  - `is_vtol_tailsitter`
  - `vehicle_type`
  - `in_transition_mode`
  - old fallback: `is_rotary_wing`
- VTOL colors come from `config_tables.vtol_modes_table`; the VTOL overlay is
  drawn as a lower screen band with `fill_alpha=0.09`.

Changed parameter labels come from `ulog.changed_parameters`, except replay logs
are skipped because they can contain many parameter changes.

Logging dropouts come from `ulog.dropouts`. Each dropout becomes a red shaded
rectangle with `fill_alpha=0.15` from `dropout.timestamp` to
`dropout.timestamp + duration_ms * 1000`.

Most time-series plots use an initial x range from
`ulog.start_timestamp - 5% duration` to `ulog.last_timestamp + 5% duration`.

## Default Log Plot Catalog

The table below lists the normal `/plot_app?log=<id>` plots created by
`configured_plots.generate_plots()`.

| Plot | Topic(s) | Field(s) / Signal(s) | Processing |
| --- | --- | --- | --- |
| Local Position 2D | `vehicle_local_position`, `vehicle_local_position_setpoint`, `vehicle_local_position_groundtruth`, `vehicle_gps_position`, `position_setpoint_triplet` | `vehicle_local_position.y` vs `x` as `Estimated`; `vehicle_local_position_setpoint.y` vs `x` as `Setpoint`; `vehicle_local_position_groundtruth.y` vs `x` as `Groundtruth`; GPS lat/lon as `GPS (projected)`; mission setpoint lat/lon as `Position Setpoints` | 2D local XY plot. GPS and mission setpoints are projected into local meters. Mission position setpoints are circles. Initial x/y scaling comes from the `Estimated` trace. `Estimated` uses `colors2[0]`, `Setpoint` uses `colors2[1]`, GPS uses `plot_config['maps_line_color']`, and mission setpoints use `plot_config['mission_setpoint_color']`. |
| Leaflet map | `vehicle_gps_position`, `vehicle_status` | GPS lat/lon, `fix_type`, `timestamp`; `vehicle_status.nav_state` | Generated as template data by `ulog_to_polyline()`. Uses only `fix_type > 2`, throttles path points to about 10 Hz, colors segments by flight mode. |
| Altitude Estimate | `vehicle_gps_position`, `vehicle_air_data` or `sensor_combined`, `vehicle_global_position`, `position_setpoint_triplet` | GPS altitude; `baro_alt_meter`; `vehicle_global_position.alt`; `position_setpoint_triplet.current.alt` | GPS altitude uses `altitude_msl_m` for new logs or `alt * 0.001` for old logs. Setpoint plotted as circles. |
| Roll/Pitch/Yaw Angle | `vehicle_attitude`, `vehicle_attitude_setpoint`, `vehicle_attitude_groundtruth` | `roll`, `pitch`, `yaw`; setpoint `roll_d`, `pitch_d`, `yaw_d`; `yaw_sp_move_rate`; groundtruth `roll`, `pitch`, `yaw` | Radians converted to degrees. Setpoints use step lines. Tailsitter VTOL logs use converted attitude from `vtol_tailsitter.tailsitter_orientation()`. |
| Roll/Pitch/Yaw Angular Rate | `vehicle_angular_velocity` or old `vehicle_attitude`, `vehicle_rates_setpoint`, `rate_ctrl_status`, groundtruth rate topic | New: `xyz[0]`, `xyz[1]`, `xyz[2]`; old: `rollspeed`, `pitchspeed`, `yawspeed`; setpoint `roll`, `pitch`, `yaw`; integrator `rollspeed_integ`, `pitchspeed_integ`, `yawspeed_integ` | Rates converted from rad/s to deg/s. Rate integrator multiplied by 100. Integral limit label comes from `MC_RR_INT_LIM`, `MC_PR_INT_LIM`, or `MC_YR_INT_LIM` if present. |
| Local Position X/Y/Z | `vehicle_local_position`, `vehicle_local_position_setpoint` | `x`, `y`, `z` and setpoint `x`, `y`, `z` | Estimated and setpoint per axis. Setpoints use step lines. NaNs can be marked. |
| Velocity | `vehicle_local_position`, `vehicle_local_position_setpoint` | `vx`, `vy`, `vz` and setpoint `vx`, `vy`, `vz` | Estimated and setpoint velocity. |
| Visual Odometry Position | `vehicle_visual_odometry`, `vehicle_local_position_groundtruth` | `x`, `y`, `z`; groundtruth `x`, `y`, `z` | Only shown if `vehicle_visual_odometry` exists. NaNs can be marked. |
| Visual Odometry Velocity | `vehicle_visual_odometry`, `vehicle_local_position_groundtruth` | `vx`, `vy`, `vz`; groundtruth `vx`, `vy`, `vz` | Only shown if visual odometry exists. |
| Visual Odometry Attitude | `vehicle_visual_odometry`, `vehicle_attitude_groundtruth` | `roll`, `pitch`, `yaw` | Radians converted to degrees. |
| Visual Odometry Attitude Rate | `vehicle_visual_odometry`, rate groundtruth topic | `rollspeed`, `pitchspeed`, `yawspeed`; groundtruth rate fields | Radians/s converted to degrees/s. |
| Visual Odometry Latency | `vehicle_visual_odometry` | `timestamp`, `timestamp_sample` | Derived latency: `1e-3 * (timestamp - timestamp_sample)`, in ms. |
| Airspeed | `vehicle_global_position`, `airspeed_validated` or `airspeed`, `vehicle_gps_position`, `tecs_status` | `vel_n`, `vel_e`; `true_airspeed_m_s` or `true_ground_minus_wind_m_s`; old `indicated_airspeed_m_s`; `vehicle_gps_position.vel_m_s`; `tecs_status.true_airspeed_sp` | Estimated groundspeed is `sqrt(vel_n^2 + vel_e^2)`. Uses `airspeed_validated` when present, otherwise old `airspeed`. |
| TECS | `tecs_status` | `height_rate`, `height_rate_setpoint` | Fixed-wing/VTOL energy controller height-rate trace. |
| Manual Control Inputs | `manual_control_setpoint`, `manual_control_switches` or old `manual_control_setpoint` | New: `roll`, `pitch`, `yaw`, `throttle`, `aux1`, `aux2`; old: `y`, `x`, `r`, `z`; switches: `mode_slot`, `kill_switch` | Old logs remap stick fields. `mode_slot` is divided by 6. `kill_switch` is converted to boolean `== 1`. Y range is fixed to `[-1.1, 1.1]`. |
| Raw Radio Control Inputs | `rc_channels` | `channels[0..N]`, `channel_count` | Fallback when `manual_control_setpoint` is absent. Plots up to 8 channels. Channel labels can use `px4_ulog.get_configured_rc_input_names(i)`. |
| Actuator Controls | Dynamic allocation: `vehicle_torque_setpoint`, `vehicle_thrust_setpoint`; old allocation: `actuator_controls_0` | Dynamic torque: `xyz[0..2]`; dynamic thrust vector `xyz`; old torque: `control[0..2]`; old thrust: `control[3]` | `ActuatorControls` abstracts old/new topics. Dynamic thrust norm is `sqrt(x^2+y^2+z^2)`. Upward thrust is `-xyz[2]`; forward thrust is `xyz[0]`. |
| Actuator Controls FFT | Same as Actuator Controls torque topic | Torque axes | FFT amplitude. Marks `MC_DTERM_CUTOFF`, `IMU_DGYRO_CUTOFF`, and `IMU_GYRO_CUTOFF` parameter frequencies when present. |
| Angular Velocity FFT | `vehicle_angular_velocity` | `xyz[0]`, `xyz[1]`, `xyz[2]` | FFT amplitude. Marks `IMU_GYRO_CUTOFF` and positive `IMU_GYRO_NF_FREQ`. |
| Angular Acceleration FFT | `vehicle_angular_acceleration` | `xyz[0]`, `xyz[1]`, `xyz[2]` | FFT amplitude. Marks `IMU_DGYRO_CUTOFF` and positive `IMU_GYRO_NF_FREQ`. |
| Actuator Controls 1 | Dynamic allocation instance 1 or old `actuator_controls_1` | Torque axes plus forward thrust | Intended for VTOL fixed-wing-mode controls. Dynamic thrust for nonzero instance is resampled to the target thrust timestamp array. |
| Motor Outputs | `actuator_motors` | `control[0..N]` | Dynamic-control-allocation logs only. Stops when a control field is missing or all NaN. |
| Servo Outputs | `actuator_servos` | `control[0..N]` | Dynamic-control-allocation logs only. Same selection logic as motor outputs. |
| Actuator Outputs Main/AUX/EXTRA | `actuator_outputs` instances 0, 1, 2 | `output[0..N]`, `noutputs` | Old logs only. Plots up to 16 outputs, but only if at least one output is not constant. Y range is fixed to `[-1, 1]`. |
| Motor RPM | `esc_status` | `esc_count`, `esc[i].esc_rpm` | Plots each ESC RPM if field exists and max RPM is greater than 0.001. |
| Raw Acceleration | `sensor_combined` | `accelerometer_m_s2[0..2]` | Direct m/s^2 axes. |
| Vibration Metrics | `vehicle_imu_status` instances 0..3 | `accel_vibration_metric` | Adds green/orange/red background zones with limits 4.905 and 9.81 m/s^2. |
| Acceleration Power Spectral Density | `sensor_combined` | `accelerometer_m_s2[0..2]` | Spectrogram PSD, summed across X/Y/Z, rendered in dB. Requires >= 100 Hz. |
| Angular Velocity Power Spectral Density | `vehicle_angular_velocity` | `xyz[0..2]` | Spectrogram PSD, summed across roll/pitch/yaw rates. Requires >= 100 Hz. |
| Angular Acceleration Power Spectral Density | `vehicle_angular_acceleration` | `xyz[0..2]` | Spectrogram PSD. Requires >= 100 Hz. |
| Raw Angular Speed (Gyroscope) | `sensor_combined` | `gyro_rad[0..2]` | Converts rad/s to deg/s. |
| Raw Acceleration FIFO | `sensor_accel_fifo` instances 0..2, virtualized as `sensor_accel_fifo_virtual` | FIFO source: `timestamp_sample`, `dt`, `samples`, `scale`, `x[s]`, `y[s]`, `z[s]`; virtual fields: `timestamp`, `x`, `y`, `z` | `add_virtual_fifo_topic_data()` expands FIFO batches into individual samples and applies `scale`. |
| Acceleration PSD FIFO | `sensor_accel_fifo_virtual` | `x`, `y`, `z` | Spectrogram PSD on expanded FIFO samples. |
| Sampling Regularity FIFO Accel | `sensor_accel_fifo` | `timestamp` | Derived `np.diff(timestamp)` plus dropout rectangles. |
| Raw Gyro FIFO | `sensor_gyro_fifo` instances 0..2, virtualized as `sensor_gyro_fifo_virtual` | FIFO source: `timestamp_sample`, `dt`, `samples`, `scale`, `x[s]`, `y[s]`, `z[s]`; virtual fields: `timestamp`, `x`, `y`, `z` | Expands FIFO batches. The plot also applies rad/s to deg/s lambdas. |
| Gyro PSD FIFO | `sensor_gyro_fifo_virtual` | `x`, `y`, `z` | Spectrogram PSD on expanded FIFO samples. |
| Raw Magnetic Field Strength | `vehicle_magnetometer` or old `sensor_combined` | `magnetometer_ga[0..2]` | Direct gauss axes. |
| Distance Sensor | `distance_sensor`, `vehicle_local_position` | `current_distance`, `variance`, `dist_bottom`, `dist_bottom_valid` | Combines rangefinder data with estimator bottom-distance estimate. |
| GPS Uncertainty | `vehicle_gps_position` | `eph`, `epv`, `hdop`, `vdop`, `s_variance_m_s`, `satellites_used`, `fix_type` | Y range is fixed to `[0, 40]` for readability. |
| GPS Noise & Jamming | `vehicle_gps_position` | `noise_per_ms`, `jamming_indicator` | Direct fields. |
| Thrust and Magnetic Field | `vehicle_magnetometer` or old `sensor_combined`, actuator thrust topic(s) | Magnetic norm from `magnetometer_ga[0..2]`; thrust from `ActuatorControls` | Magnetic norm is `sqrt(mx^2+my^2+mz^2)`. Thrust is dynamic thrust norm or old `control[3]`. |
| Power | `battery_status`, `system_power` | `voltage_v`, `current_a`, `discharged_mah`, `remaining`, optional `ocv_estimate`, `internal_resistance_estimate`, `voltage5v_v`, `sensors3v3[0]` | `discharged_mah` is divided by 100. `remaining` is multiplied by 10. Internal resistance is multiplied by 1000 to mOhm. |
| Temperature | `sensor_baro`, `sensor_accel`, `airspeed`, `battery_status`, `esc_status` | `temperature`, `air_temperature_celsius`, `esc[i].esc_temperature` | ESC temperature plotted per ESC if present and nonzero. |
| Estimator Flags | `estimator_status` | `health_flags`, `timeout_flags`, `innovation_check_flags` | Decodes selected bits from `innovation_check_flags`. Plots only nonzero flags, max 8 lines. If none are nonzero, plots health flags so absence is not ambiguous. |
| Failsafe Flags | `vehicle_status`, `failsafe_flags` | `vehicle_status.failsafe`, `failsafe_and_user_took_over`, all nonzero fields from `failsafe_flags` except `timestamp` and `mode_req_*` | Skips always-set `auto_mission_missing` and `offboard_control_signal_lost`. Field names are converted from underscores to spaces for legends. |
| CPU & RAM | `cpuload` | `ram_usage`, `load` | Adds horizontal mean spans for both `load` and `ram_usage`. Y range is fixed to `[0, 1]`. |
| Sampling Regularity of Sensor Data | `sensor_combined`, `estimator_status` | `sensor_combined.timestamp`, `estimator_status.time_slip` | Derived `np.diff(sensor_combined.timestamp)` plus `time_slip * 1e6`. Adds dropout rectangles. |

After these plots, Flight Review appends:

- A changed-parameters table from `ulog.initial_parameters`,
  default-parameter data, and `ulog.changed_parameters`.
- A logged messages table from ULog logged messages and events.
- Optional boot console, process, and performance-counter text from
  `ulog.msg_info_multiple_dict`.

## Compatibility Rules

`configured_plots.py` contains several compatibility branches:

- Barometer and magnetometer topic:
  - New logs: `vehicle_air_data`, `vehicle_magnetometer`
  - Old logs: `sensor_combined`
- GPS altitude:
  - New logs with `ver_data_format >= 2`: `altitude_msl_m`
  - Old logs: `alt * 0.001`
- Angular rates:
  - New logs: `vehicle_angular_velocity.xyz[0..2]`
  - Old logs: `vehicle_attitude.rollspeed`, `pitchspeed`, `yawspeed`
- Manual control:
  - New logs: `manual_control_setpoint.roll/pitch/yaw/throttle`
  - Old logs: `manual_control_setpoint.y/x/r/z`
- System power:
  - Old `voltage5V_v`, `voltage3V3_v`, or `voltage3v3_v` fields are renamed
    to current `voltage5v_v` and `sensors3v3[0]`.
- TECS:
  - Old `tecs_status.airspeed_sp` is renamed to `true_airspeed_sp`.
- Actuators:
  - Dynamic control allocation uses `actuator_motors`, `actuator_servos`,
    `vehicle_torque_setpoint`, and `vehicle_thrust_setpoint`.
  - Old allocation uses `actuator_controls_0`, `actuator_controls_1`, and
    `actuator_outputs`.

## PID Analysis Page

The PID page is requested with `?plots=pid_analysis&log=<id>`.

`pid_analysis_plots.get_pid_analysis_plots()` builds:

- Regular angular-rate plots for roll, pitch, yaw.
- PID step-response plots for rate axes.
- Optional PID step-response plots for roll and pitch attitude.

Required rate-analysis topics and fields:

- Rate topic:
  - New: `vehicle_angular_velocity.xyz[0..2]`
  - Old fallback: `rate_ctrl_status.rollspeed/pitchspeed/yawspeed`
- Setpoint topic:
  - `vehicle_rates_setpoint.roll`
  - `vehicle_rates_setpoint.pitch`
  - `vehicle_rates_setpoint.yaw`
- Actuator thrust:
  - New dynamic allocation: `vehicle_thrust_setpoint.xyz`
  - Old allocation: `actuator_controls_0.control[3]`
- Rate integrator:
  - `rate_ctrl_status.rollspeed_integ`
  - `rate_ctrl_status.pitchspeed_integ`
  - `rate_ctrl_status.yawspeed_integ`

Optional attitude-analysis topics and fields:

- `vehicle_attitude.roll`, `vehicle_attitude.pitch`
- `vehicle_attitude_setpoint.roll_d`, `vehicle_attitude_setpoint.pitch_d`

Processing:

- Thrust is resampled to the gyro or attitude timestamp array with
  `scipy.interpolate.interp1d()`.
- Rates and attitude are converted from radians to degrees.
- `pid_analysis.Trace` equalizes samples to a uniform time grid.
- It windows the time series, applies a Hanning window, performs Wiener
  deconvolution between setpoint input and measured output, and computes average
  response curves plus 2D response histograms.
- `plot_pid_response()` renders the histogram as a Bokeh image and overlays the
  low-rate response curve. If high-rate input exists, it overlays a separate
  high-rate response curve.

## Statistics Page

The statistics page is requested with `/stats` or `/plot_app?stats=1`. These are
not per-log ULog plots. They come from SQLite:

- `Logs`
- `LogsGenerated`

`StatisticsPlots` loads:

- total number of logs
- CI log count
- public/private upload counts grouped into 6-hour intervals
- recent public logs from the last 90 days
- generated metadata such as duration, autostart ID, hardware, UUID, software
  version, and flight-mode durations

Statistics plots include:

- Number of log files on server
- Public board flight hours
- Public board flight count
- Public unique boards
- Public airframe flight count
- Public firmware-version flight count
- Public flight-mode hours

Most statistics plots are stacked area plots. `plot_groups_as_stack()`:

- Groups data by day.
- Applies cumulative sums for most plots.
- Limits displayed groups to 20 and combines smaller groups as `Others`.
- Uses `bokeh.figure.varea_stack()`.

## Overview Images

`overview_generator.py` generates static PNG map previews with Matplotlib and
Smopy:

- Loads the ULog through `load_ulog_file()`.
- Reads `vehicle_gps_position`.
- Filters `fix_type > 2`.
- Extracts lat/lon through `get_lat_lon_alt_deg()`.
- Chooses a Smopy map zoom that stays under a tile limit.
- Draws the GPS path in red and saves `<log_id>.png`.

This is separate from the interactive Bokeh plots on the main page.

## Implications for This Agent

For our flight-log analysis agent, the most reusable pieces are:

- Use a topic allowlist or lazy topic loading to avoid parsing everything when
  only a few plots are needed.
- Normalize old/new PX4 field names before plotting or metric extraction.
- Keep plot definitions declarative: title, source topic, fields, transforms,
  y-axis label, and optional overlays.
- Use `timestamp` as the canonical x-axis, with `timestamp_sample` for raw sensor
  frequency analysis when present.
- Reuse common overlays:
  - flight mode bands from `vehicle_status.nav_state`
  - VTOL mode bands from `vehicle_status`
  - parameter-change markers from `ulog.changed_parameters`
  - dropout markers from `ulog.dropouts`
- For high-rate signals, provide downsampling and optional FFT/PSD tools instead
  of pushing all raw samples into the frontend.
