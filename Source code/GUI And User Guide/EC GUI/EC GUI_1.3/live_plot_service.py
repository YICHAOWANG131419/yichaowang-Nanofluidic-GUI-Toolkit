# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:47:46 2025

@author: p81942ai
"""

# live_plot_service.py
# Threaded live plotting that never blocks acquisition timing.
# - Producer (acquisition) calls push(index, current) WITHOUT blocking.
# - A plot thread drains a queue and schedules Tk-safe draws via root.after(0, ...).
# - Toggle on/off at runtime with set_enabled(True/False).

from collections import deque
import threading, time, queue

class LivePlotService:

    def set_target(self, ax, canvas):
        """Switch the axis/canvas used for drawing."""
        self.ax = ax
        self.canvas = canvas
        self.clear()          # reset buffers/line so we rebuild cleanly
    def __init__(self, root, ax, canvas, maxlen=10000, max_fps=20, manage_axis_text=True):

        self.root = root
        self.ax = ax
        self.canvas = canvas
        self.manage_axis_text = manage_axis_text

        self.enabled = False
        self.q = queue.Queue(maxsize=10000)      # big enough; push is non-blocking
        self.buf_x = deque(maxlen=maxlen)
        self.buf_y = deque(maxlen=maxlen)

        self._stop = threading.Event()
        self._thread = None
        self._line = None

        self._last_draw = 0.0
        self._draw_interval = 1.0 / float(max_fps)

    # ---- lifecycle ----
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=1.0)
        self._thread = None

    def clear(self):
        self.buf_x.clear()
        self.buf_y.clear()
        with self.q.mutex:
            self.q.queue.clear()
        self._line = None
        # do not draw here; next draw will rebuild line

    # ---- control ----
    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        if not self.enabled:
            # flush queue quickly so producer timing stays identical
            with self.q.mutex:
                self.q.queue.clear()

    # ---- producer side (acquisition thread) ----
    def push(self, idx: int, current: float):
        if not self.enabled:
            return
        try:
            self.q.put_nowait((idx, float(current)))
        except queue.Full:
            # drop silently to preserve acquisition timing
            pass

    # ---- plot thread ----
    def _run(self):
        while not self._stop.is_set():
            if not self.enabled:
                time.sleep(0.02)
                continue

            drained = 0
            try:
                while True:
                    i, c = self.q.get_nowait()
                    self.buf_x.append(i)
                    self.buf_y.append(c)
                    drained += 1
            except queue.Empty:
                pass

            now = time.perf_counter()
            if drained and (now - self._last_draw) >= self._draw_interval:
                self._last_draw = now
                # Schedule a Tk-safe draw in the main/UI thread
                try:
                    self.root.after(0, self._draw_once)
                except Exception:
                    # root may be closing; ignore
                    pass

            time.sleep(0.01)

    # ---- main-thread drawing ----
    def _draw_once(self):
        if self._line is None:
            (self._line,) = self.ax.plot(list(self.buf_x), list(self.buf_y),
                                         marker=".", linewidth=1, label="I (A)")
            if self.manage_axis_text:
                self.ax.set_title("Live: I vs Sample Index")
                self.ax.set_xlabel("Sample Index")
                self.ax.set_ylabel("Current (A)")
                self.ax.grid(True)
            # Avoid duplicate legends if host axis already has items:
            try:
                handles, labels = self.ax.get_legend_handles_labels()
                if labels:
                    self.ax.legend()
            except Exception:
                pass
        else:
            self._line.set_data(list(self.buf_x), list(self.buf_y))
            self.ax.relim()
            self.ax.autoscale_view()

        try:
            self.canvas.draw_idle()
        except Exception:
            pass
