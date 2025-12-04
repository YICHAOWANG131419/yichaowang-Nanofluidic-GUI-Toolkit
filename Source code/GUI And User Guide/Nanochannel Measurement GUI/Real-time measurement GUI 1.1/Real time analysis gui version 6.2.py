# ===================================================Import required libraries =======================================================
# ===== Standard libraries =====
import os     # For handling file paths, directories, and file operations
import csv    # For reading and writing parameter settings in CSV format
import time   # For delays, timing measurements, and real-time control loops

# ===== Scientific libraries =====
import numpy as np  # Numerical calculations, arrays, and waveform generation
import pandas as pd  # Data processing and exporting tables to CSV
import matplotlib.pyplot as plt  # For generating external plots (e.g., final analysis)
from matplotlib.figure import Figure  # Create Matplotlib figures without auto-show
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed Matplotlib plots into Tkinter GUI
import mplcursors  # Add interactive data-point hover cursors to plots
from threading import Thread
from queue import Queue, Empty
# ===== GUI libraries =====
import tkinter as tk  # Base GUI components (windows, labels, buttons, etc.)
from tkinter import ttk, filedialog, messagebox  # ttk: themed widgets; filedialog: open/save file dialogs; messagebox: pop-up alerts
from time import perf_counter
from live_plot_service import LivePlotService
from iv_live_window import LivePlotWindow
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
# EC-GUI style live plot globals
live_svc = None
live_win = None
analysis_win = None
analysis_canvas = None
# ===========================Check for PyVISA availability to determine if real instrument control is possible =======================
try:
    import pyvisa
    VISA_AVAILABLE = True  # VISA is available, real device communication enabled
except ModuleNotFoundError:
    print("[Warning] PyVISA not installed. Running in offline mode.")  # Fallback to simulation mode
    VISA_AVAILABLE = False



# Flag to control measurement loop
stop_flag = False  # If set to True, measurement will stop immediately

gpib_var = None  # Global variable to hold the GPIB address selection


# --- live-plot thread plumbing ---
plot_queue = Queue()
plot_after_id = None
_plot_v, _plot_i = [], []  # live series for the embedded plot



# ===================================double thread =================================================
_last_draw = 0.0
_draw_count = 0

measurement_paused = False
resume_index = 0
keithley = None
# ---- Thread-safe bridge to Tk mainloop ----


_tk_queue = Queue()

def tk_post(fn, *args, **kwargs):
    """Schedule fn(*args, **kwargs) to run on the Tk (main) thread."""
    _tk_queue.put((fn, args, kwargs))

def _tk_pump():
    """Drain the queue on the Tk thread; reschedule itself."""
    try:
        while True:
            fn, args, kwargs = _tk_queue.get_nowait()
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print("[Tk callback error]", e)
    except Empty:
        pass
    root.after(15, _tk_pump)   # ~60 FPS pump
def run_iv_test_threaded(from_start=True):
    use_lp = (live_plot_var.get() == "On")
    # Live plot UI must be created on the MAIN thread
    if use_lp:
        _ensure_live_window()           # <-- main thread only
        if live_svc:
            live_svc.set_enabled(True)
    Thread(target=run_iv_test, args=(from_start, use_lp), daemon=True).start()

def continue_measurement():
    global stop_flag, measurement_paused
    if measurement_paused:
        stop_flag = False
        run_iv_test_threaded(from_start=False)

def restart_measurement():
    global stop_flag, measurement_paused, resume_index
    stop_flag = False
    measurement_paused = False
    resume_index = 0
    run_iv_test_threaded(from_start=True)

#=========================================================== Keithley control functions ===============================================
# Functions to stop the ongoing IV measurement
def stop_measurement():
    global stop_flag, measurement_paused
    stop_flag = True
    measurement_paused = True  # paused, not restart

def render_final_plot(v, i, pattern, device_label, seg_lengths):
    """
    Show the final IV analysis in a Tk Toplevel.
    Keeps your save_plot/save_data globals working.
    Adds:
      - FP/DP/FN/DN colored segments (Triangle)
      - Linear fit (I = G*V + b) with G, R, V@I=0, R^2
      - Zero lines and optional V@I=0 marker
      - Clean grid, legend, and hover tooltips
    """
    global fig, analysis_win, analysis_canvas

    # Close any previous analysis window
    try:
        if analysis_win is not None and analysis_win.winfo_exists():
            analysis_win.destroy()
    except Exception:
        pass

    # New window
    analysis_win = tk.Toplevel(root)
    analysis_win.title(f"IV Analysis — {device_label}")
    analysis_win.geometry("900x540")

    # Figure & axes
    fig = Figure(figsize=(9, 5.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"I–V Curve — Device {device_label}")

    # ---------- plot data ----------
    v = np.asarray(v).ravel()
    i = np.asarray(i).ravel()

    # Segment colors (Triangle)
    seg_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    seg_labels = ["FP", "DP", "FN", "DN"]

    if pattern == "Triangle" and seg_lengths is not None and len(v) >= 2:
        L = np.array(seg_lengths, dtype=int)
        # Guard: truncate if lengths overshoot
        L = np.minimum(L, len(v))
        s1 = L[0]
        s2 = s1 + L[1]
        s3 = s2 + L[2]
        s4 = s3 + L[3]
        slices = [(0, s1), (s1, s2), (s2, s3), (s3, min(s4, len(v)))]
        for (a, b_), c, lbl in zip(slices, seg_colors, seg_labels):
            if b_ > a:
                ax.plot(v[a:b_], i[a:b_],
                        color=c, linestyle='-', marker='.', markersize=2.5, linewidth=1.2, label=lbl)

    else:
        # Sine: segment by sign(V) and slope sign to mimic FP/DP/FN/DN
        if len(v) >= 3:
            seg_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]  # FP, DP, FN, DN
            color_map = {"FP": seg_colors[0], "DP": seg_colors[1],
                         "FN": seg_colors[2], "DN": seg_colors[3]}
            shown = set()

            eps = 1e-12
            dv = np.diff(v)

            # sign of dv with epsilon; forward-fill zeros so flat segments keep prior direction
            sgn_dv = np.where(dv > eps, 1, np.where(dv < -eps, -1, 0))
            for k in range(1, len(sgn_dv)):
                if sgn_dv[k] == 0:
                    sgn_dv[k] = sgn_dv[k - 1]
            if sgn_dv[0] == 0:
                sgn_dv[0] = 1 if (v[-1] - v[0]) >= 0 else -1  # fallback

            def _sign(x):
                return 1 if x > eps else (-1 if x < -eps else 0)

            def _label(vsign, dvsign):
                # FP: rising in +V; DP: falling in +V; FN: falling in –V; DN: rising in –V
                if dvsign > 0 and vsign >= 0: return "FP"
                if dvsign < 0 and vsign >= 0: return "DP"
                if dvsign < 0 and vsign <= 0: return "FN"
                return "DN"  # dvsign > 0 and vsign <= 0

            # build contiguous segments with same label
            start = 0
            prev_lbl = _label(_sign(v[1]), sgn_dv[0])
            for k in range(1, len(sgn_dv)):
                lbl = _label(_sign(v[k + 1]), sgn_dv[k])
                if lbl != prev_lbl:
                    a = start
                    b_ = k + 1
                    ax.plot(v[a:b_], i[a:b_],
                            color=color_map[prev_lbl], linestyle='-',
                            marker='.', markersize=2.5, linewidth=1.2,
                            label=(prev_lbl if prev_lbl not in shown else None))
                    shown.add(prev_lbl)
                    start = k
                    prev_lbl = lbl
            # last chunk
            ax.plot(v[start:], i[start:],
                    color=color_map[prev_lbl], linestyle='-',
                    marker='.', markersize=2.5, linewidth=1.2,
                    label=(prev_lbl if prev_lbl not in shown else None))

    # Zero reference lines
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)

    # ---------- linear analysis ----------
    G = np.nan
    R = np.nan
    V_zero = np.nan
    R2 = np.nan
    if len(v) >= 2 and np.all(np.isfinite(v)) and np.all(np.isfinite(i)):
        try:
            m, b = np.polyfit(v, i, 1)  # I = m*V + b
            G = m
            R = (1.0 / m) if m != 0 else np.nan
            V_zero = (-b / m) if m != 0 else np.nan

            # R^2
            i_fit = m * v + b
            ss_res = float(np.sum((i - i_fit) ** 2))
            ss_tot = float(np.sum((i - np.mean(i)) ** 2)) if len(i) > 1 else 0.0
            R2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            # Fit line over data span
            vmin, vmax = float(np.min(v)), float(np.max(v))
            v_fit = np.array([vmin, vmax])
            i_fit_end = m * v_fit + b
            ax.plot(v_fit, i_fit_end, color="k", linestyle="--", linewidth=1.1, label="Linear fit")

            # Mark V@I=0 if inside span
            if np.isfinite(V_zero) and (vmin <= V_zero <= vmax):
                ax.axvline(V_zero, color="k", linestyle=":", lw=1.0)
                ax.plot([V_zero], [0.0], "ko", ms=3)
        except Exception:
            pass

    # Legend
    try:
        ax.legend(loc="upper right")
    except Exception:
        pass

    # Info textbox
    txt = []
    if np.isfinite(G):   txt.append(f"G = {G:.3e} S")
    if np.isfinite(R):   txt.append(f"R = {R:.3e} Ω")
    if np.isfinite(V_zero): txt.append(f"V@I=0 = {V_zero:.4g} V")
    if np.isfinite(R2):  txt.append(f"R² = {R2:.4f}")
    if txt:
        ax.text(0.02, 0.98, "\n".join(txt),
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"))

    fig.tight_layout()

    # Hover tooltips
    try:
        mplcursors.cursor(ax.lines, hover=True).connect(
            "add",
            lambda sel: sel.annotation.set_text(f"V = {sel.target[0]:.6g} V\nI = {sel.target[1]:.6g} A")
        )
    except Exception:
        pass

    # Embed in Tk
    analysis_canvas = FigureCanvasTkAgg(fig, master=analysis_win)
    analysis_canvas.draw()
    analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Toolbar (optional)
    try:
        NavigationToolbar2Tk(analysis_canvas, analysis_win)
    except Exception:
        pass

    # Safe close
    def _close_analysis():
        try:
            safe_close_plot(analysis_canvas, fig)
        except Exception:
            pass
        try:
            analysis_win.destroy()
        except Exception:
            pass

    analysis_win.protocol("WM_DELETE_WINDOW", _close_analysis)



def _ensure_live_window():
    """Create/prepare the popup live I–V window and (re)attach the plot service."""
    global live_win, live_svc

    # Create the popup window if needed
    if live_win is None or not live_win.winfo_exists():
        live_win = LivePlotWindow(root)
        live_win.protocol("WM_DELETE_WINDOW", _on_live_window_close)

    ax = live_win.ax

    # 1) Clear + style axes (before wiring the service)
    ax.cla()
    try:
        ax.grid(True, linestyle=":", alpha=0.6)
    except Exception:
        ax.grid(True)
    ax.set_title("Live: I vs V")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")

    # 2) Create or retarget the background plotting service
    if live_svc is None:
        live_svc = LivePlotService(
            root, ax, live_win.canvas,
            maxlen=10000,
            max_fps=20,
            manage_axis_text=False,  # we set labels above
            xmode='v'                # x = Voltage, y = Current
        )
        live_svc.start()
    else:
        live_svc.manage_axis_text = False
        live_svc.set_target(ax, live_win.canvas)
        live_svc.clear()

    # 3) Dotted markers + thin line
    try:
        live_svc.set_style(
            marker='.', markersize=3,
            linestyle='-', linewidth=1.4,
            label='I (A)'
        )
    except Exception:
        pass

    # 4) Enable the service and request a draw
    try:
        live_svc.set_enabled(True)
    except Exception:
        pass

    # NOTE: do NOT call ax.legend() here; the service adds it once the line exists
    try:
        live_win.canvas.draw_idle()
    except Exception:
        pass


def _on_live_window_close():
    """Safety: stop acquisition, zero V, output OFF, close window, stop live service."""
    global stop_flag, live_win
    stop_flag = True
    # Best-effort safety write using current channel/address
    try:
        if VISA_AVAILABLE:
            rm = pyvisa.ResourceManager()
            addr = current_gpib_address()
            ch = channel_var.get() if 'channel_var' in globals() else 'smua'
            if addr:
                inst = rm.open_resource(addr)
                inst.write_termination = '\n'; inst.read_termination = '\n'
                inst.timeout = 5000
                try:
                    inst.write(f"{ch}.source.levelv=0")
                    inst.write(f"{ch}.source.output = {ch}.OUTPUT_OFF")
                finally:
                    inst.close()
    except Exception:
        pass
    try:
        if live_svc:
            live_svc.set_enabled(False)
            live_svc.stop()  # stop the background plot thread
    except Exception:
        pass
    try:
        if live_win and live_win.winfo_exists(): live_win.destroy()
    finally:
        live_win = None

# Main function to run the IV test using Keithley device (EC-GUI live plot edition)
def run_iv_test(from_start=True, use_live_plot=False):
    global stop_flag, measurement_paused, v, i, fig  # v,i used by save_data()
    stop_flag = False
    measurement_paused = False

    # === Retrieve user input parameters ===
    step        = float(entry_step.get())
    N           = int(entry_repeat.get())
    cycles      = int(entry_cycles.get())
    pattern     = pattern_combo.get()           # "Triangle" or "Sine"
    device_label = entry_device.get()
    sine_steps  = int(entry_sine_steps.get())
    sine_dwell  = float(entry_sine_dwell.get())

    channel     = channel_var.get()             # "smua" / "smub"

    # New bounds
    start_at = float(entry_start.get())
    end_at   = float(entry_end.get())
    pos_max  = float(entry_pos_max.get())
    neg_max  = float(entry_neg_max.get())

    # For sine magnitude
    amp = max(abs(pos_max - start_at), abs(start_at - neg_max))

    # Persist settings
    settings_to_save = {
        "channel": channel,
        "step_size": entry_step.get(),
        "start_at": entry_start.get(),
        "end_at": entry_end.get(),
        "pos_max": entry_pos_max.get(),
        "neg_max": entry_neg_max.get(),
        "repeat_number": entry_repeat.get(),
        "cycle_number": entry_cycles.get(),
        "device_label": entry_device.get(),
        "current_limit": current_limit_var.get(),
        "range_mode": range_mode_var.get(),
        "voltage_range": entry_rangev.get(),
        "current_range": entry_rangei.get(),
        "voltage_pattern": pattern,
        "gpib_address": gpib_var.get(),
        "sine_steps_per_cycle": entry_sine_steps.get(),
        "sine_dwell_s": entry_sine_dwell.get(),
    }
    save_iv_settings_csv(settings_to_save)

    # === Build waveform ===
    if pattern == "Triangle":
        n_fp = max(1, int(round(abs(pos_max - start_at) / step)))    # FP: start -> pos_max
        n_dp = n_fp                                                  # DP: pos_max -> start
        n_fn = max(1, int(round(abs(start_at - neg_max) / step)))    # FN: start -> neg_max
        n_dn = max(1, int(round(abs(end_at   - neg_max) / step)))    # DN: neg_max -> end_at

        up_pos   = np.linspace(start_at, pos_max, n_fp, endpoint=True)
        down_pos = np.linspace(pos_max,  start_at, n_dp, endpoint=True)
        down_neg = np.linspace(start_at, neg_max,  n_fn, endpoint=True)
        up_neg   = np.linspace(neg_max,  end_at,   n_dn, endpoint=True)

        one_leg      = np.concatenate([up_pos, down_pos, down_neg, up_neg])
        one_cycle    = np.repeat(one_leg, N)
        voltage_steps = np.tile(one_cycle, cycles)

        seg_lengths = np.array([n_fp, n_dp, n_fn, n_dn]) * N

    else:
        # --- Pure sine: exactly sine_steps * cycles samples, no extra tail ---
        points = max(2, sine_steps)                     # samples per cycle
        total_points = points * max(1, int(cycles))     # total samples
        phase = np.linspace(0.0, 2.0*np.pi*cycles, total_points, endpoint=False)
        voltage_steps = start_at + amp * np.sin(phase)

        seg_lengths = None

        # Debug print of nominal frequency if dwell > 0
        try:
            if sine_dwell > 0:
                freq = 1.0 / (points * sine_dwell)
                print(f"[Sine] steps/cycle={points}, dwell={sine_dwell:.6f}s  ->  f≈{freq:.6f} Hz")
            else:
                print(f"[Sine] steps/cycle={points}, dwell=0 -> using instrument latency per step")
        except Exception:
            pass


    # === Initialize result lists ===
    global voltage_readings, current_readings, t_f, resume_index
    if from_start:
        voltage_readings = []
        current_readings = []
        t_f = []
        resume_index = 0
        start_clock = perf_counter()
    else:
        start_clock = perf_counter() - (t_f[-1] if t_f else 0.0)

    # >>> pure acquisition-time accumulator <<<
    t_meas_cum = 0.0 if from_start else (float(t_f[-1]) if t_f else 0.0)

    # === Prepare EC-GUI live plot ===
    if use_live_plot:
        def _prep_live_ui():
            _ensure_live_window()
            if live_svc:
                live_svc.set_enabled(True)
        root.after(0, _prep_live_ui)

    # === Measurement ===
    points_captured = 0
    if VISA_AVAILABLE:
        keithley = None
        try:
            rm = pyvisa.ResourceManager()
            gpib_address = current_gpib_address()
            if not gpib_address:
                root.after(0, lambda: messagebox.showerror(
                    "GPIB Address", "Please select or type a GPIB address."))
                return

            keithley = rm.open_resource(gpib_address)
            keithley.write_termination = '\n'
            keithley.read_termination  = '\n'
            keithley.timeout = 10000

            # TSP config
            keithley.write(f"{channel}.reset()")
            keithley.write(f"{channel}.nvbuffer1.clear()")
            keithley.write(f"{channel}.source.limiti = {current_limit_var.get()}")
            if range_mode_var.get() == "Auto":
                keithley.write(f"{channel}.measure.autorangei = {channel}.AUTORANGE_ON")
                keithley.write(f"{channel}.source.autorangev = {channel}.AUTORANGE_ON")
            else:
                keithley.write(f"{channel}.measure.autorangei = {channel}.AUTORANGE_OFF")
                keithley.write(f"{channel}.source.autorangev = {channel}.AUTORANGE_OFF")
                keithley.write(f"{channel}.measure.rangei = {entry_rangei.get()}")
                keithley.write(f"{channel}.source.rangev  = {entry_rangev.get()}")

            keithley.write(f"{channel}.source.output = {channel}.OUTPUT_ON")
            keithley.write(f"{channel}.measure.nplc = 1")

            def measure_iv():
                keithley.write(f"ireading, vreading = {channel}.measure.iv()")
                try:
                    resp = keithley.query("print(ireading, vreading)")
                except Exception:
                    resp = keithley.query("printnumber(ireading,vreading)")
                p = resp.replace('\t', ',').split(",")
                return float(p[1]), float(p[0])  # (V, I)

            auto_min = 0.01 if range_mode_var.get() == "Auto" else 0.0
            target_step_time = max(0.0, sine_dwell) if pattern == "Sine" else auto_min

            for idx in range(resume_index, len(voltage_steps)):
                if stop_flag:
                    resume_index = idx
                    break

                step_t0 = perf_counter()
                vset = float(voltage_steps[idx])
                keithley.write(f"{channel}.source.levelv={vset}")

                # ---- measure & account only acquisition time ----
                meas_t0 = perf_counter()
                v_read, i_read = measure_iv()
                meas_t1 = perf_counter()

                #t_meas_cum += (meas_t1 - meas_t0)   # << only measurement time
                # t_f.append(t_meas_cum)
                # Wall-clock since the run started (includes set-level + bus overhead, not UI)
                t_f.append(perf_counter() - start_clock)

                voltage_readings.append(v_read)
                current_readings.append(i_read)
                points_captured += 1

                if use_live_plot and (live_svc is not None):
                    try:
                        live_svc.push(v_read, i_read)  # x=V, y=I
                    except Exception:
                        pass

                elapsed = perf_counter() - step_t0
                remain = target_step_time - elapsed
                if remain > 0:
                    time.sleep(remain)

        except Exception as e:
            root.after(0, lambda e=e: messagebox.showerror(
                "Keithley Error", f"Device connection failed:\n{e}"))
            return
        finally:
            try:
                if keithley is not None:
                    keithley.write(f"{channel}.source.levelv=0")
                    keithley.write(f"{channel}.source.output = {channel}.OUTPUT_OFF")
                    keithley.close()
            except Exception:
                pass

    else:
        # Offline simulation (always produce some data)
        current_readings = (voltage_steps * 1e-6) if pattern == "Triangle" else (np.sin(voltage_steps*np.pi) * 1e-6)
        voltage_readings = voltage_steps
        # Simulate acquisition-time spacing (e.g., 1 ms per point)
        t_step = max(1e-3, sine_dwell if pattern == "Sine" else 1e-3)
        t_meas_cum_base = 0.0 if from_start else (float(t_f[-1]) if t_f else 0.0)
        t_f.extend(list(t_meas_cum_base + np.arange(1, len(voltage_steps)+1) * t_step))
        points_captured = len(voltage_steps)
        if use_live_plot and (live_svc is not None):
            for vv, ii in zip(voltage_readings, current_readings):
                live_svc.push(vv, ii)

    # Stop live service but keep the window
    if use_live_plot:
        root.after(0, lambda: (live_svc and live_svc.set_enabled(False)))

    # Nothing captured? Tell user and bail gracefully
    if points_captured == 0:
        root.after(0, lambda: messagebox.showwarning(
            "No Data", "No data points were recorded. Check GPIB address, ranges, or limits."))
        return

    # === Final arrays & final plot ===
    v = np.array(voltage_readings)
    i = np.array(current_readings)
    root.after(0, lambda: render_final_plot(v, i, pattern, device_label, seg_lengths))




def current_gpib_address():
    try:
        # if it's a tk.StringVar, read it; otherwise coerce to string
        return gpib_var.get() if isinstance(gpib_var, tk.StringVar) else str(gpib_var)
    except Exception:
        return ""
# Main function to execute the IV (current-voltage) test using a Keithley sourcemeter.
def show_iv_control():
    """
    Main function to execute the IV (current-voltage) test using a Keithley sourcemeter.

    This function performs the following operations:

    1. Retrieves user-defined test parameters from the GUI, including:
       - Voltage step size
       - Voltage amplitude (maximum applied voltage)
       - Number of sweep repetitions
       - Voltage waveform pattern (Triangle or Sine)
       - Current limit, range mode (Auto or Manual), and device label
       - Measurement channel (smua/smub)
       - GPIB address and range values (manual mode)

    2. Saves the current parameters to a temporary CSV file for backup/reference.

    3. Generates the appropriate voltage waveform array based on the selected pattern.
       - Triangle: 4 segments per cycle (0→+A→0→–A→0)
       - Sine: Single cycle waveform from sin(x) scaled to the amplitude

    4. If PyVISA is available and the Keithley device is connected:
       - Initializes the device and clears its buffer
       - Applies current limit, range settings, and enables output
       - Iteratively steps through voltage values, recording measured (V, I)
       - Optionally updates a live embedded plot during acquisition
       - Handles user interrupt with graceful stop and output shutoff

    5. If PyVISA is not available, simulates (V, I) values offline for debugging.

    6. After measurement:
       - Finalizes output shutdown
       - Generates a final matplotlib IV plot showing each quadrant sweep
       - Supports mplcursors interaction for hover tooltips on data points

    This is the central control function for all IV acquisition,
    combining GUI interaction, device communication, live plotting, and result visualization.
    """

    global entry_step,  entry_repeat, entry_cycles, entry_start,entry_end,entry_neg_max,entry_pos_max
    global entry_device, entry_rangei, entry_rangev
    global current_limit_var, range_mode_var
    global pattern_combo, live_plot_var, channel_var, gpib_var


    live_plot_var = tk.StringVar(value="Off") # Default is Off
    
    # === Function to save GPIB address to local file ===
    def save_gpib_address():
        """
        Save the current GPIB address selected by the user into a local file.
        Also update the dropdown list if it's a new address.
        """
        address = gpib_var.get()
        if address:  # Only save non-empty address
            with open("gpib_address.txt", "w") as f:
                f.write(address)
                print(f"[Saved GPIB Address] {address}")
        
        # If it's a new address, insert into the dropdown options    
        if address not in gpib_combo["values"]:
                current_values = list(gpib_combo["values"])
                current_values.insert(0, address)
                gpib_combo["values"] = current_values

    # === Clear previous widgets from right_frame ===
    for widget in right_frame.winfo_children():
        widget.destroy()

    # === Title label for IV control panel ===
    tk.Label(right_frame, text="IV Control Panel", font=("Arial", 14, "bold")).pack(pady=10)

    form_frame = tk.Frame(right_frame)
    form_frame.pack(pady=5)
    # === Channel Selection Dropdown ===
    row_channel = tk.Frame(form_frame)
    row_channel.pack(pady=2)
    tk.Label(row_channel, text="Channel:", width=20, anchor="w").pack(side=tk.LEFT)

    global channel_var
    channel_var = tk.StringVar(value="smub")  # Default to Channel B
    channel_combo = ttk.Combobox(row_channel, textvariable=channel_var, width=17)
    channel_combo["values"] = ["smua", "smub"]
    channel_combo.pack(side=tk.LEFT)
    
    # === Frame for all control buttons (Run, Stop, Save, etc.) ===
    global control_frame
    control_frame = tk.Frame(right_frame)
    control_frame.pack(pady=10)
    
    def make_entry(parent, label, default):
        """
        Create a labeled entry row inside a parent Tkinter widget.
        
        Parameters:
            parent (tk.Widget): The container frame to embed the row into
            label (str): The text label shown to the left of the entry field
            default (any): The default value to pre-fill in the entry box

        Returns:
            tk.Entry: The Entry widget created, for later access to its value
        """
        row = tk.Frame(parent)  # Create a horizontal row frame
        row.pack(pady=2)        # Add vertical spacing between rows
        tk.Label(row, text=label, width=20, anchor="w").pack(side=tk.LEFT) # Left-aligned label with fixed width
        entry = tk.Entry(row, width=20)  # Input field
        entry.insert(0, str(default))    # Insert default value as string
        entry.pack(side=tk.LEFT)         # Pack to the right of the label

        return entry  # Return reference for later value acces
    
    # === Input fields for IV test parameters ===
    entry_step = make_entry(form_frame, "Step Size (V):", 0.05)
    
    # New voltage limits & endpoints
    entry_start = make_entry(form_frame, "Start at (V):", 0.0)
    entry_end   = make_entry(form_frame, "End at (V):", 0.0)
    entry_pos_max = make_entry(form_frame, "Positive Max (V):", 1.0)
    entry_neg_max = make_entry(form_frame, "Negative Max (V):", -1.0)
    
    entry_repeat = make_entry(form_frame, "Repeat Number:", 1)
    entry_cycles = make_entry(form_frame, "Number of Cycles:", 1)
    globals()["entry_cycles"] = entry_cycles
    entry_device = make_entry(form_frame, "Device Label:", "1_XXXX")


    # === Current Limit Dropdown ===
    row_limit = tk.Frame(form_frame)
    row_limit.pack(pady=2)
    tk.Label(row_limit, text="Current Limit (A):", width=20, anchor="w").pack(side=tk.LEFT)

    current_limit_var = tk.StringVar(value="1e-3")
    current_limit_combo = ttk.Combobox(row_limit, textvariable=current_limit_var, width=17)
    current_limit_combo["values"] = ["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8", "1e-9"]
    current_limit_combo.pack(side=tk.LEFT)

    # === Range Mode Dropdown ===
    row_range_mode = tk.Frame(form_frame)
    row_range_mode.pack(pady=2)
    tk.Label(row_range_mode, text="Range Mode:", width=20, anchor="w").pack(side=tk.LEFT)

    range_mode_var = tk.StringVar(value="Auto")
    range_mode_combo = ttk.Combobox(row_range_mode, textvariable=range_mode_var, width=17)
    range_mode_combo["values"] = ["Auto", "Manual"]
    range_mode_combo.pack(side=tk.LEFT)

    # === Manual Voltage & Current Range Entry ===
    row_rangev = tk.Frame(form_frame)
    row_rangev.pack(pady=2)
    tk.Label(row_rangev, text="Voltage Range (V):", width=20, anchor="w").pack(side=tk.LEFT)
    entry_rangev = tk.Entry(row_rangev, width=20)
    entry_rangev.insert(0, "1")  # default 1 V
    entry_rangev.pack(side=tk.LEFT)

    row_rangei = tk.Frame(form_frame)
    row_rangei.pack(pady=2)
    tk.Label(row_rangei, text="Current Range (A):", width=20, anchor="w").pack(side=tk.LEFT)
    entry_rangei = tk.Entry(row_rangei, width=20)
    entry_rangei.insert(0, "1e-3")  # default 1 mA
    entry_rangei.pack(side=tk.LEFT)
    # --- Sine-only options ---
    global entry_sine_steps, entry_sine_dwell
    
    sine_box = tk.LabelFrame(form_frame, text="Sine Options")
    sine_box.pack(pady=4, fill=tk.X)
    
    entry_sine_steps = make_entry(sine_box, "Steps per Cycle:", 200)       # integer
    entry_sine_dwell = make_entry(sine_box, "Dwell per Step (s):", 0.005)  # seconds (e.g., 5 ms)
    # === Pattern ComboBox ===
    row_pattern = tk.Frame(form_frame)
    row_pattern.pack(pady=2)
    tk.Label(row_pattern, text="Voltage Pattern:", width=20, anchor="w").pack(side=tk.LEFT)

    global pattern_combo
    pattern_combo = ttk.Combobox(row_pattern, values=["Triangle", "Sine"], width=17)
    pattern_combo.current(0)
    pattern_combo.pack(side=tk.LEFT)

    # === GPIB Address ===
    row_gpib = tk.Frame(form_frame)
    row_gpib.pack(pady=2)

    tk.Label(row_gpib, text="GPIB Address:", width=20, anchor="w").pack(side=tk.LEFT)
    
    global gpib_var #Global it
    gpib_var = tk.StringVar()

    # === Initialize default list of GPIB addresses (can be extended) ===
    if VISA_AVAILABLE:
        try:
            rm = pyvisa.ResourceManager()
            address_list = list(rm.list_resources())  # Dynamically detect available GPIB devices
        except Exception as e:
            print(f"[GPIB Error] Could not list resources: {e}")
            address_list = []
    else: # Offline test
        # Fallback: default address list or empty
        address_list = ["GPIB0::20::INSTR", "GPIB0::24::INSTR"]


    # === Try to load the last saved GPIB address from file ===
    if os.path.exists("gpib_address.txt"):
        try:
            with open("gpib_address.txt", "r") as f:
                saved_address = f.read().strip() # Read and remove trailing newline
                if saved_address and saved_address not in address_list:
                    address_list.insert(0, saved_address) # Insert at top if not already in list
        except Exception as e:
            print(f"[Error reading saved GPIB address] {e}") # Handle any file access or decode error gracefully

    gpib_combo = ttk.Combobox(row_gpib, textvariable=gpib_var, width=17, state="readonly")
    gpib_combo["values"] = address_list
    
    if address_list:
        gpib_var.set(address_list[0])   # ✅ seed the StringVar
        gpib_combo.current(0)           # keep the UI in sync
    else:
        gpib_var.set("")                # nothing detected; allow typing

    gpib_combo.pack(side=tk.LEFT, padx=5)
    
    # ✅ ensure variable updates when user picks a different item
    gpib_combo.bind("<<ComboboxSelected>>", lambda e: gpib_var.set(gpib_combo.get()))

    # === Save button for storing currently selected GPIB address ===
    tk.Button(row_gpib, text="💾 Save GPIB Address", command=save_gpib_address, width=20).pack(side=tk.LEFT, padx=5)
    
    # === Automatically load previously saved IV configuration (from last session) ===
    saved = load_iv_settings_csv()
    if saved:
        channel_var.set(saved.get("channel", "smub"))
        entry_step.delete(0, tk.END); entry_step.insert(0, saved.get("step_size", "0.05"))

        entry_repeat.delete(0, tk.END); entry_repeat.insert(0, saved.get("repeat_number", "1"))
        entry_device.delete(0, tk.END); entry_device.insert(0, saved.get("device_label", "1_XXXX"))
        current_limit_var.set(saved.get("current_limit", "1e-3"))
        range_mode_var.set(saved.get("range_mode", "Auto"))
        entry_rangev.delete(0, tk.END); entry_rangev.insert(0, saved.get("voltage_range", "1"))
        entry_rangei.delete(0, tk.END); entry_rangei.insert(0, saved.get("current_range", "1e-3"))
        pattern_combo.set(saved.get("voltage_pattern", "Triangle"))
        # New fields (fallbacks keep old files working)
        entry_start.delete(0, tk.END);   entry_start.insert(0, saved.get("start_at", "0"))
        entry_end.delete(0, tk.END);     entry_end.insert(0, saved.get("end_at", "0"))
        entry_pos_max.delete(0, tk.END); entry_pos_max.insert(0, saved.get("pos_max", saved.get("amplitude", "1.0")))
        entry_neg_max.delete(0, tk.END); entry_neg_max.insert(0, saved.get("neg_max", str(-float(saved.get("amplitude", "1.0")))))
        entry_sine_steps.delete(0, tk.END); entry_sine_steps.insert(0, saved.get("sine_steps_per_cycle", "200"))
        entry_sine_dwell.delete(0, tk.END); entry_sine_dwell.insert(0, saved.get("sine_dwell_s", "0.005"))


        # Only set GPIB address if the saved one is in the available dropdown list
        saved_gpib = saved.get("gpib_address", "")
        if saved_gpib and saved_gpib in gpib_combo["values"]:
            gpib_var.set(saved_gpib)

     # === Section for saving/loading/deleting named parameter configurations ===
    config_frame = tk.Frame(right_frame)
    config_frame.pack(pady=8)
    
    # Function to save parameter configuration
    def save_config_interactively():
        settings = {
    "channel": channel_var.get(),
    "step_size": entry_step.get(),
    "repeat_number": entry_repeat.get(),
    "cycle_number": entry_cycles.get(),
    "device_label": entry_device.get(),
    "current_limit": current_limit_var.get(),
    "range_mode": range_mode_var.get(),
    "voltage_range": entry_rangev.get(),
    "current_range": entry_rangei.get(),
    "voltage_pattern": pattern_combo.get(),
    "gpib_address": gpib_var.get(),
    "start_at": entry_start.get(),
    "end_at": entry_end.get(),
    "pos_max": entry_pos_max.get(),
    "neg_max": entry_neg_max.get(),
    "sine_steps_per_cycle": entry_sine_steps.get(),
    "sine_dwell_s": entry_sine_dwell.get()
}
        save_named_iv_config_interactive(settings)


    def update_config_list():
        """
        Update the dropdown list of configurations.
        Combines both files in configs/ and valid entries from config_history.txt.
        Removes entries that no longer exist.
        """
        config_names = []

        # Add files from configs/ folder
        if os.path.exists(CONFIG_DIR):
            for fname in os.listdir(CONFIG_DIR):
                if fname.endswith(".csv"):
                    config_names.append(fname[:-4])  # remove ".csv"

        # Add names from config history if file still exists
        if os.path.exists(CONFIG_HISTORY_FILE):
            try:
                with open(CONFIG_HISTORY_FILE, "r") as f:
                    for line in f:
                        path = line.strip()
                        if os.path.exists(path):
                            name = os.path.basename(path)
                            if name.endswith(".csv"):
                                config_names.append(name[:-4])
            except Exception as e:
                print(f"[History Read Error] {e}")

        # Remove duplicates and sort
        config_names = sorted(set(config_names))
        config_combo["values"] = config_names
        config_var.set("")  # Reset selection

    
    
    def apply_iv_settings(settings):
        """
        Apply a settings dictionary to all GUI input fields.
        """
        channel_var.set(settings.get("channel", "smub"))
        
        entry_step.delete(0, tk.END)
        entry_step.insert(0, settings.get("step_size", "0.05"))
        


        entry_repeat.delete(0, tk.END)
        entry_repeat.insert(0, settings.get("repeat_number", "1"))

        entry_cycles.delete(0, tk.END)  
        entry_cycles.insert(0, settings.get("cycle_number", "1"))
        
        entry_device.delete(0, tk.END)
        entry_device.insert(0, settings.get("device_label", "1_XXXX"))
        
        current_limit_var.set(settings.get("current_limit", "1e-3"))
        range_mode_var.set(settings.get("range_mode", "Auto"))
        
        entry_rangev.delete(0, tk.END)
        entry_rangev.insert(0, settings.get("voltage_range", "1"))
        
        entry_rangei.delete(0, tk.END)
        entry_rangei.insert(0, settings.get("current_range", "1e-3"))
            
        pattern_combo.set(settings.get("voltage_pattern", "Triangle"))
        gpib_var.set(settings.get("gpib_address", "GPIB0::21::INSTR"))

        entry_start.delete(0, tk.END)   
        entry_start.insert(0, settings.get("start_at", "0"))
        
        entry_end.delete(0, tk.END)
        entry_end.insert(0, settings.get("end_at", "0"))
        
        entry_pos_max.delete(0, tk.END)
        entry_pos_max.insert(0, settings.get("pos_max", "1.0"))
        
        entry_neg_max.delete(0, tk.END)
        entry_neg_max.insert(0, settings.get("neg_max", "-1.0"))

        entry_sine_steps.delete(0, tk.END)
        entry_sine_steps.insert(0, settings.get("sine_steps_per_cycle", "200"))
        
        entry_sine_dwell.delete(0, tk.END)
        entry_sine_dwell.insert(0, settings.get("sine_dwell_s", "0.005"))


    # Function to load saved configurations
    def load_config_callback():
        """
        Load a configuration file using file dialog and apply values to input fields.
        Also record the file path in config_history.txt and update the dropdown list.
        """
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Load IV Configuration"
        )
        if not file_path:
            return  

        settings = load_named_iv_config_by_path(file_path)
        if settings:
            apply_iv_settings(settings)  

            add_to_config_history(file_path)  

            filename_only = os.path.basename(file_path)
            current_values = list(config_combo["values"])
            if filename_only not in current_values:
                current_values.append(filename_only)
                config_combo["values"] = current_values
            config_combo.set(filename_only)

    def load_from_history():
        """
        Load settings from a configuration selected in the dropdown.
        Path is retrieved from config_history.txt.
        """
        selected = config_combo.get()
        if not selected:
            return
        try:
            with open("config_history.txt", "r") as f:
                for line in f:
                    if selected in os.path.basename(line.strip()):
                        file_path = line.strip()
                        settings = load_named_iv_config_by_path(file_path)
                        if settings:
                            apply_iv_settings(settings)
                            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")
    
    def remove_from_config_history(filepath):
        """
        Remove a specific path from config_history.txt
        """
        if not os.path.exists(CONFIG_HISTORY_FILE):
            return
        try:
            with open(CONFIG_HISTORY_FILE, "r") as f:
                lines = f.readlines()
            with open(CONFIG_HISTORY_FILE, "w") as f:
                for line in lines:
                    if line.strip() != filepath:
                        f.write(line)
        except Exception as e:
            print(f"[Remove Error] Failed to update config history: {e}")

    # Function to delete selected configuration
    def delete_config():
        """
        Delete the selected configuration file from disk.
        Works for both configs/xxx.csv and full path entries from history.
        Also removes it from the config history if applicable.
        """
        selected = config_var.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a configuration to delete.")
            return


        full_paths = load_config_history()
        full_match = None
        for p in full_paths:
            if os.path.basename(p) == selected:
                full_match = p
                break


        if full_match and os.path.exists(full_match):
            confirm = messagebox.askyesno("Delete Confirmation", f"Delete '{selected}' from:\n{full_match}?")
            if confirm:
                try:
                    os.remove(full_match)
                    messagebox.showinfo("Deleted", f"Deleted configuration:\n{full_match}")
                    remove_from_config_history(full_match)
                    update_config_list()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete:\n{e}")
            return


        file_path = os.path.join(CONFIG_DIR, f"{selected}.csv")
        if os.path.exists(file_path):
            confirm = messagebox.askyesno("Delete Confirmation", f"Delete '{selected}' from:\n{file_path}?")
            if confirm:
                try:
                    os.remove(file_path)
                    messagebox.showinfo("Deleted", f"Deleted configuration:\n{file_path}")
                    update_config_list()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete:\n{e}")
        else:
            messagebox.showerror("Error", "Configuration file not found.")

    
    # === Save Button ===
    tk.Button(config_frame, text="💾 Save Parameter Configuration", command=save_config_interactively).pack(side=tk.LEFT, padx=5)
    # === Dropdown for selecting existing configs ===
    config_var = tk.StringVar()
    config_combo = ttk.Combobox(config_frame, textvariable=config_var, state="readonly", width=30)
    config_combo.pack(side=tk.LEFT, padx=5)
    # === Load & Delete Buttons ===
    tk.Button(config_frame, text="🔄 Load Selected Configuration", command=load_config_callback).pack(side=tk.LEFT, padx=5)
    tk.Button(config_frame, text="🧾 Load History Configuration", command=load_from_history).pack(side=tk.LEFT, padx=5)
    tk.Button(config_frame, text="❌ Delete Selected Configuration", command=delete_config).pack(side=tk.LEFT, padx=5)
    # === Initial list update ===
    update_config_list()
    # === Save Data & Plot Buttons Below Configuration Section ===
    save_btn_frame = tk.Frame(right_frame)
    save_btn_frame.pack(pady=8)

    tk.Button(save_btn_frame, text="📉 Save Plot", command=save_plot, width=20).pack(side=tk.LEFT, padx=20)
    tk.Button(save_btn_frame, text="📊 Save Data", command=save_data, width=20).pack(side=tk.LEFT, padx=20)

  


    # === Create a frame for embedded plots on the right side ===
    global fig_frame
    fig_frame = tk.Frame(right_frame)
    fig_frame.pack(pady=10, fill=tk.BOTH, expand=True)
    
    # ✅ Load last IV settings from file and apply to GUI fields
    settings = load_iv_settings_csv()
    if settings:
        apply_iv_settings(settings)
    

    
    
    # === Function to create bottom run/stop/live plotting buttons (moved up)
    def create_run_controls():
        run_and_plot_frame = tk.Frame(right_frame)
        run_and_plot_frame.pack(before=save_btn_frame, pady=10)

        tk.Button(run_and_plot_frame, text="🔁 Clear Buffer", command=clear_buffer, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(run_and_plot_frame, text="▶ Run IV Test",
          command=lambda: run_iv_test_threaded(True), width=15).pack(side=tk.LEFT, padx=5)
       
        tk.Button(run_and_plot_frame, text="⏸ Continue",
                  command=continue_measurement, width=15).pack(side=tk.LEFT, padx=5)

        tk.Button(run_and_plot_frame, text="🔁 Restart", command=restart_measurement, width=15).pack(side=tk.LEFT, padx=5)

        tk.Button(run_and_plot_frame, text="🛑 Stop", command=stop_measurement, width=15).pack(side=tk.LEFT, padx=5)

        tk.Label(run_and_plot_frame, text="Live Plotting:", font=("Arial", 11)).pack(side=tk.LEFT, padx=5)
        live_plot_combo = ttk.Combobox(run_and_plot_frame, textvariable=live_plot_var, width=6, state="readonly")
        live_plot_combo["values"] = ["Off", "On"]
        live_plot_combo.pack(side=tk.LEFT)

    # Function: Clear Keithley buffer manually
    def clear_buffer():
        try:
            if VISA_AVAILABLE:
                rm = pyvisa.ResourceManager()
                addr = current_gpib_address()
                if not addr:
                    messagebox.showerror("GPIB Address", "Please select or type a GPIB address.")
                    return
                keithley = rm.open_resource(addr)
                keithley.write_termination = '\n'
                keithley.read_termination  = '\n'
                keithley.timeout = 5000
                channel = channel_var.get()
                keithley.write(f"{channel}.nvbuffer1.clear()")
                keithley.close()
                print("[Buffer Cleared]")
            else:
                print("[Offline Mode] No hardware to clear buffer.")
        except Exception as e:
            messagebox.showerror("Buffer Clear Error", f"Failed to clear buffer:\n{e}")

  
    # Remove any leftover widgets from control panel area (e.g. from previous run)
    for widget in control_frame.winfo_children():
        widget.destroy()
    
    create_run_controls()
  
# ======================== Safe shutdown helpers ========================

def safe_close_inst(inst):
    """安全关闭 Keithley/VISA 资源（支持 TSP 或 SCPI）"""
    if inst is None:
        return
    try:
        # 先尽量让仪器退出运行状态（TSP）
        for cmd in ('abort', 'errorqueue.clear()', 'reset()'):
            try:
                inst.write(cmd + '\n')
            except Exception:
                pass
        # SCPI 清理（若在 SCPI）
        try:
            inst.write('*CLS\n')
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            inst.close()
        except Exception:
            pass


# 你在程序里哪里创建了 after 计时器，就把 id 保存到这个列表，如：
# aid = root.after(100, some_func); _after_ids.append(aid)
_after_ids = []

def cancel_all_afters(root):
    global _after_ids
    for aid in list(_after_ids):
        try:
            root.after_cancel(aid)
        except Exception:
            pass
    _after_ids = []


def safe_close_plot(canvas=None, fig=None):
    """
    Safely destroy a FigureCanvasTkAgg (if provided) and close its Matplotlib figure.
    This function is idempotent and never raises.
    """
    # Destroy the Tk widget that hosts the canvas
    try:
        if canvas is not None:
            try:
                widget = canvas.get_tk_widget()
            except Exception:
                widget = None
            if widget is not None:
                try:
                    widget.destroy()
                except Exception:
                    pass
            # If a figure wasn't passed, try to close the one attached to the canvas
            try:
                attached_fig = getattr(canvas, "figure", None)
            except Exception:
                attached_fig = None
            if fig is None and attached_fig is not None:
                try:
                    plt.close(attached_fig)
                except Exception:
                    pass
    except Exception:
        pass

    # Close an explicitly provided figure
    try:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
    except Exception:
        pass



# ======================== Main window close callback ========================
def on_close():
    """
    Safe shutdown:
      - confirm
      - stop acquisition
      - stop/disable EC live plot + close its window (if present)
      - cancel Tk after jobs (if helper exists)
      - zero and turn outputs OFF (TSP first, SCPI fallback)
      - close VISA sessions
      - destroy root
    """
    global stop_flag, measurement_paused, keithley, live_svc, live_win

    # Confirm
    try:
        ans = messagebox.askyesno("Exit Confirmation", "Are you sure to exit and disable output?")
    except Exception:
        ans = True
    if not ans:
        return

    # Stop acquisition flags
    try: stop_flag = True
    except Exception: pass
    try: measurement_paused = True
    except Exception: pass

    # Stop EC-GUI live plot service + window (if present)
    svc = globals().get('live_svc')
    if svc:
        try: svc.set_enabled(False)
        except Exception: pass
        try: svc.stop()
        except Exception: pass

    win = globals().get('live_win')
    try:
        if win and getattr(win, "winfo_exists", lambda: False)():
            win.destroy()
    except Exception:
        pass
    live_win = None

    # (Legacy) embedded live plot helpers are optional—call only if they exist
    stop_lp = globals().get('stop_live_plotter')
    if callable(stop_lp):
        try: stop_lp()
        except Exception: pass

    safe_close = globals().get('safe_close_plot')
    if callable(safe_close):
        lc = globals().get('live_canvas')
        lf = globals().get('live_fig')
        try: safe_close(lc, lf)
        except Exception: pass

    cancel_afters = globals().get('cancel_all_afters')
    if callable(cancel_afters):
        try: cancel_afters(root)
        except Exception: pass

    # Helper: robustly zero & turn outputs off on any VISA session
    def _safety_zero(inst):
        # TSP (2600 series): both channels
        for s in ("smua", "smub"):
            try:
                inst.write(f"{s}.source.levelv = 0")
            except Exception:
                pass
            try:
                inst.write(f"{s}.source.output = {s}.OUTPUT_OFF")
            except Exception:
                pass
        # SCPI fallback (harmless on TSP if accepted)
        try:
            inst.write(":SOUR:VOLT 0")
        except Exception:
            pass
        try:
            inst.write(":OUTP OFF")
        except Exception:
            pass

    # VISA safety writes (import pyvisa here to satisfy linters)
    try:
        if globals().get('VISA_AVAILABLE'):
            import pyvisa  # local import avoids 'undefined name' warnings
            rm = pyvisa.ResourceManager()

            addr = None
            get_addr = globals().get('current_gpib_address')
            if callable(get_addr):
                try: addr = get_addr()
                except Exception: addr = None

            # Try by address
            if addr:
                inst1 = None
                try:
                    inst1 = rm.open_resource(addr)
                    inst1.write_termination = '\n'
                    inst1.read_termination  = '\n'
                    inst1.timeout = 5000
                    _safety_zero(inst1)
                except Exception:
                    pass
                finally:
                    try:
                        if inst1 is not None:
                            inst1.close()
                    except Exception:
                        pass

            # Also act on any already-open global handle
            if keithley is not None:
                try:
                    _safety_zero(keithley)
                except Exception:
                    pass
                finally:
                    try:
                        keithley.close()
                    except Exception:
                        pass
                    keithley = None

            print("[Safe Exit] Outputs OFF and level=0 issued.")
    except Exception as e:
        print(f"[Close Error] Safety shutdown issue: {e}")

    # Destroy window
    try:
        root.destroy()
    except Exception:
        pass



# =================================================== Save Functions ================================================
# === Global Save Functions for IV Plot and Data ===
# Save plot function
def save_plot():
    """
    Open a file dialog and save the current IV matplotlib plot (global 'fig') as a PNG image.

    - Opens a file save dialog for the user to specify the output path.
    - Saves the global figure (fig) with 300 DPI resolution.
    - Displays an error message if saving fails.
    """
    global fig
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save IV Plot as Image"
        )
        if file_path:
            fig.savefig(file_path, dpi=600)
            print(f"[Saved Plot] {file_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save plot:\n{e}")

# Save data function (XLSX: V, I, time)
def save_data():
    """
    Save the recorded IV data to an Excel .xlsx file with columns:
    V (voltage), I (current), time (s), in that order.
    Falls back to CSV if an Excel writer engine is not available.
    """
    global t_f, v, i
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            title="Save IV Data"
        )
        if not file_path:
            return  # user cancelled

        # Ensure same length & correct order: V, I, time
        n = min(len(v), len(i), len(t_f))
        df = pd.DataFrame({
            "V": np.asarray(v)[:n],
            "I": np.asarray(i)[:n],
            "time": np.asarray(t_f)[:n],
        })

        try:
            # Requires openpyxl or xlsxwriter installed
            df.to_excel(file_path, index=False)
            print(f"[Saved Data] {file_path}")
        except ImportError:
            # Fallback to CSV with same column order
            alt = os.path.splitext(file_path)[0] + ".csv"
            df.to_csv(alt, index=False)
            messagebox.showwarning(
                "Excel writer not found",
                f"openpyxl/xlsxwriter not installed.\nSaved CSV instead:\n{alt}"
            )
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data:\n{e}")

    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save data:\n{e}")


# === Temporary Backup for Last Test Parameters ===
SETTINGS_CSV = "last_iv_settings.csv"

# Save parameters
def save_iv_settings_csv(settings_dict):
    """
    Save the latest IV test settings to a temporary CSV file ('last_iv_settings.csv').
    - Accepts a dictionary with parameter-value pairs.
    - Overwrites the existing CSV file each time.
    - Used for quick access to the last-run configuration.
    """
    try:
        with open(SETTINGS_CSV, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key, value in settings_dict.items():
                writer.writerow([key, value])
    except Exception as e:
        print(f"[Save Error] Failed to save settings: {e}")

# Load functions
def load_iv_settings_csv():
    """
    Load previously saved IV test settings from 'last_iv_settings.csv'.
    - Returns a dictionary with parameter-value pairs.
    - Skips header and handles missing file or read errors.
    """
    if not os.path.exists(SETTINGS_CSV):
        return None
    settings = {}
    try:
        with open(SETTINGS_CSV, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    settings[row[0]] = row[1]
        return settings
    except Exception as e:
        print(f"[Load Error] Failed to load settings: {e}")
        return None

# === Named Configuration Save/Load/Delete ===
CONFIG_DIR = "iv_configs"

# Check if 'configs' already exists as a file
if os.path.isfile(CONFIG_DIR):
    raise OSError(f"A file named '{CONFIG_DIR}' already exists. Please delete or rename it.")

# Make 'configs/' directory safely
os.makedirs(CONFIG_DIR, exist_ok=True)


# === History file for storing loaded config paths ===
CONFIG_HISTORY_FILE = "config_history.txt"

def add_to_config_history(filepath):
    """
    Add filepath to history if not already present.
    Avoids duplication.
    """
    if not os.path.exists(CONFIG_HISTORY_FILE):
        with open(CONFIG_HISTORY_FILE, "w") as f:
            f.write(filepath + "\n")
        return

    with open(CONFIG_HISTORY_FILE, "r") as f:
        paths = f.read().splitlines()
    if filepath not in paths:
        with open(CONFIG_HISTORY_FILE, "a") as f:
            f.write(filepath + "\n")

def load_config_history():
    """
    Load list of previously used config paths.
    """
    if not os.path.exists(CONFIG_HISTORY_FILE):
        return []
    with open(CONFIG_HISTORY_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


#Save configuration with a defined name
def save_named_iv_config_interactive(settings_dict):
    """
    Save IV settings to a user-selected CSV file using a file dialog.
    """
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save IV Configuration"
        )
        if not file_path:
            return  # User canceled
        with open(file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value"])
            for key, value in settings_dict.items():
                writer.writerow([key, value])
        print(f"[Saved Config] {file_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save config:\n{e}")


# Load saved configuration from drop down list
def load_named_iv_config_interactive():
    """
    Load IV settings from a user-selected CSV file using a file dialog.
    Returns a dictionary of settings.
    """
    try:
        file_path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Load IV Configuration"
        )
        if not file_path:
            return None
        settings = {}
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    settings[row[0]] = row[1]
        print(f"[Loaded Config] {file_path}")
        add_to_config_history(file_path)
        return settings
    except Exception as e:
        messagebox.showerror("Load Error", f"Failed to load config:\n{e}")
        return None

def load_named_iv_config_by_path(file_path):
    """
    Load IV config from arbitrary file path.
    """
    if not os.path.exists(file_path):
        return None
    settings = {}
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 2:
                settings[row[0]] = row[1]
    return settings


# ==================================================== Main GUI Initialization =================================================
# Create the main application window (root)
root = tk.Tk()
root.title("Keithley I-V Control GUI")  # Set the window title
root.geometry("1200x850")               # Initial window size (only effective before .state("zoomed"))

# === [Main Layout Frame] ===
# Create a main frame to hold left and right sections
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)  # Allow the main frame to expand with window resizing

# === [Left Panel: Control Buttons] ===
# Create the left sidebar for control buttons (gray background)
left_frame = tk.Frame(main_frame, padx=10, pady=10, bg="lightgray")
left_frame.pack(side=tk.LEFT, fill=tk.Y)  # Stick to left and fill vertical space only

# === [Right Panel: Display Area (Graph, Inputs, Tables)] ===
# Create the right panel for all interactive outputs
right_frame = tk.Frame(main_frame, padx=10, pady=10)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # Fill and expand to occupy remaining space

# === [IV Control Button] ===
# Add a button on the left to launch the IV test control interface
btn_iv_control = tk.Button(
    left_frame,
    text="IV Control",
    command=show_iv_control,  # Callback to render the IV test interface on the right
    width=40                  # Width in text units (for consistent size)
)
btn_iv_control.pack(pady=4)  # Add vertical spacing between buttons

# === [Window Behavior Settings] ===

root.state("zoomed")          # Start with the window maximized (Windows only; ignored on some systems)
root.resizable(True, True)  # Disable resizing window by dragging (fixed size UI)

# Set the window close event to trigger safe shutdown sequence
# This ensures Keithley output is disabled before exiting
root.protocol("WM_DELETE_WINDOW", on_close)

# === [Start the Main Event Loop] ===
# Enter the Tkinter event loop (keeps the window open and responsive)
root.mainloop()