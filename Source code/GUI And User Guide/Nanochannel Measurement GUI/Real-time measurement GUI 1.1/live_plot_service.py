# -*- coding: utf-8 -*-
"""
Main-thread live plotting service for Tk/Matplotlib.
- Producer threads call push(x, y) (non-blocking).
- A Tk 'after' ticker on the MAIN thread drains a queue and draws
  at up to max_fps, so no Tk API is called off-thread.
"""

from collections import deque
import queue

class LivePlotService:
    def __init__(self, root, ax, canvas, *, maxlen=10000, max_fps=20,
                 xmode='v', manage_axis_text=True,
                 marker=None, linestyle='-', linewidth=1.4, markersize=0, label='I (A)'):

        self.root = root
        self.ax = ax
        self.canvas = canvas
        self.manage_axis_text = manage_axis_text

        # style
        self._style = dict(marker=marker, linestyle=linestyle,
                           linewidth=linewidth, markersize=markersize, label=label)

        # buffers and queue
        self.buf_x = deque(maxlen=int(maxlen))
        self.buf_y = deque(maxlen=int(maxlen))
        self.q = queue.Queue(maxsize=10000)  # drop when full (loss-tolerant viz)

        # ticker
        self._tick_id = None
        self._period_ms = max(5, int(1000 / max(1, max_fps)))

        # artist & legend
        self._line = None
        self._legend_shown = False

        # control
        self.enabled = False
        self.xmode = 'v' if str(xmode).lower().startswith('v') else 'index'

        # autoscale padding
        self._pad_x = 0.05
        self._pad_y = 0.10

        # ---- performance knobs (new) ----
        self.plot_tail = 1000          # draw only the most recent N points
        self.autoscale_every = 10      # do autoscale every N frames (>=1)
        self._frame_count = 0          # internal frame counter
        self.max_drain = 2000          # max samples drained per tick from queue
        self.freeze_autoscale = False  # if True, never autoscale during run

        # cosmetics
        try:
            self.ax.grid(True, linestyle=":", alpha=0.6)
        except Exception:
            pass
        if self.manage_axis_text:
            self._apply_labels()

    # ---------- lifecycle ----------
    def start(self):
        if self._tick_id is None:
            self._tick()

    def stop(self):
        if self._tick_id is not None:
            try:
                self.root.after_cancel(self._tick_id)
            except Exception:
                pass
            self._tick_id = None

    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        if not self.enabled:
            self._flush_queue()

    def clear(self):
        self.buf_x.clear(); self.buf_y.clear()
        self._flush_queue()
        try:
            if self._line is not None:
                self._line.remove()
        except Exception:
            pass
        self._line = None
        self._legend_shown = False
        self._request_draw()

    def set_target(self, ax, canvas):
        try:
            if self._line is not None:
                self._line.remove()
        except Exception:
            pass
        self.ax = ax
        self.canvas = canvas
        self._line = None
        try:
            self.ax.grid(True, linestyle=":", alpha=0.6)
        except Exception:
            pass
        if self.manage_axis_text:
            self._apply_labels()
        self._legend_shown = False
        self._request_draw()

    def set_style(self, *, marker=None, linestyle=None, linewidth=None,
                  markersize=None, label=None):
        if marker is not None:     self._style['marker'] = marker
        if linestyle is not None:  self._style['linestyle'] = linestyle
        if linewidth is not None:  self._style['linewidth'] = linewidth
        if markersize is not None: self._style['markersize'] = markersize
        if label is not None:      self._style['label'] = label
        try:
            if self._line is not None:
                self._line.set(**self._style)
        except Exception:
            pass
        self._request_draw()

    def set_xmode(self, xmode: str):
        self.xmode = 'v' if str(xmode).lower().startswith('v') else 'index'
        if self.manage_axis_text:
            self._apply_labels()
        self._request_draw()

    # ---- convenience setters (optional) ----
    def set_plot_window(self, n_points: int = 1000, autoscale_every: int = 10):
        self.plot_tail = max(100, int(n_points))
        self.autoscale_every = max(1, int(autoscale_every))

    def set_refresh_rate(self, max_fps: int = 12):
        self._period_ms = max(5, int(1000 / max(1, int(max_fps))))

    def set_freeze_autoscale(self, freeze: bool = True):
        self.freeze_autoscale = bool(freeze)

    def set_max_drain(self, n: int = 2000):
        self.max_drain = max(100, int(n))

    # ---- producer API ----
    def push(self, x, y):
        if not self.enabled:
            return
        try:
            self.q.put_nowait((float(x), float(y)))
        except Exception:
            pass  # drop when full (viz-only)

    # ---------- internals (MAIN THREAD) ----------
    def _flush_queue(self):
        try:
            with self.q.mutex:
                self.q.queue.clear()
        except Exception:
            pass

    def _apply_labels(self):
        if self.xmode == 'v':
            self.ax.set_title("Live: I vs V")
            self.ax.set_xlabel("Voltage (V)")
        else:
            self.ax.set_title("Live: I vs Sample Index")
            self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Current (A)")

    def _tick(self):
        # MAIN THREAD: drain a bounded chunk and draw
        if self.enabled:
            drained = 0
            try:
                for _ in range(int(self.max_drain)):   # <-- bounded drain
                    x, y = self.q.get_nowait()
                    self.buf_x.append(x); self.buf_y.append(y)
                    drained += 1
            except queue.Empty:
                pass

            if drained:
                self._draw_once()

        self._tick_id = self.root.after(self._period_ms, self._tick)

    def _request_draw(self):
        try:
            self.root.after_idle(self._draw_once)
        except Exception:
            pass

    def _draw_once(self):
        if self._line is None:
            (self._line,) = self.ax.plot([], [], **self._style)
            self._legend_once()

        # draw only recent window to reduce per-frame cost
        tail = int(self.plot_tail)
        ybuf = list(self.buf_y)[-tail:]
        if self.xmode == 'v':
            xbuf = list(self.buf_x)[-tail:]
        else:
            start = max(0, len(self.buf_y) - len(ybuf))
            xbuf = list(range(start, start + len(ybuf)))

        self._line.set_data(xbuf, ybuf)

        # autoscale at a reduced frequency (unless frozen)
        self._frame_count += 1
        if not self.freeze_autoscale:
            if (self._frame_count % int(self.autoscale_every)) == 0:
                self._autoscale(xbuf, ybuf)

        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _legend_once(self):
        if self._legend_shown:
            return
        try:
            handles, labels = self.ax.get_legend_handles_labels()
            if labels:
                self.ax.legend(loc='upper right')
                self._legend_shown = True
        except Exception:
            pass

    def _autoscale(self, x, y):
        if not x or not y:
            return
        xmin, xmax = (min(x), max(x))
        ymin, ymax = (min(y), max(y))

        # X
        if xmax == xmin:
            span = max(abs(xmax), 1e-12) * 0.10
            xmin, xmax = xmin - span, xmax + span
        else:
            pad = (xmax - xmin) * self._pad_x
            xmin, xmax = xmin - pad, xmax + pad

        # Y
        if ymax == ymin:
            span = max(abs(ymax), 1e-12) * 0.50
            ymin, ymax = ymin - span, ymax + span
        else:
            pad = (ymax - ymin) * self._pad_y
            ymin, ymax = ymin - pad, ymax + pad

        try:
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.ax.grid(True, linestyle=":", alpha=0.6)
        except Exception:
            pass
