# keithley_controller.py  — 2614B 修复版
import time
import numpy as np
import pyvisa
class KeithleyController:
    def __init__(self):
        self.rm = None
        self.inst = None
        self.connected = False
        self.idn = ""
        self.is_2600_tsp = False  # 2600/261x/263x 系列（TSP）
        self._abort = False       # <-- 初始化

    def stop(self):
        """Request to abort the ongoing output loop."""
        self._abort = True

    def connect(self, addr: str):
        """
        Open VISA resource and initialize using TSP only (no SCPI),
        so we don't trigger -285 'TSP Syntax error'.
        """
        self.rm = pyvisa.ResourceManager()
        self.inst = self.rm.open_resource(addr)

        # Terminations: TSP lines end with LF
        self.inst.write_termination = '\n'
        self.inst.read_termination  = '\n'
        self.inst.timeout = 20000  # ms

        # For GPIB, clear device buffer (safe no-op on others)
        try:
            self.inst.clear()
        except Exception:
            pass

        # ---- TSP-only init (no ':' or '*IDN?' here) ----
        self.inst.write('errorqueue.clear()')
        # optional: event log too
        try:
            self.inst.write('eventlog.clear()')
        except Exception:
            pass

        # Safe reset & outputs off
        self.inst.write('smua.source.output = smua.OUTPUT_OFF')
        self.inst.write('smub.source.output = smub.OUTPUT_OFF')
        self.inst.write('smua.reset()')
        self.inst.write('smub.reset()')

        # Handshake using TSP print/read (no SCPI query)
        self.inst.write('print(localnode.model)')
        model = self.inst.read().strip()
        self.inst.write('print(localnode.serialno)')
        serial = self.inst.read().strip()
        # >>> add these two lines <<<
        self.is_2600_tsp = True
        self.idn = f"{model} (S/N {serial})"
        self.connected = True
        return f"{model} (S/N {serial})"

    def disconnect(self):
        try:
            if self.inst is not None:
                # turn outputs off on exit
                try:
                    self.inst.write('smua.source.output = smua.OUTPUT_OFF')
                    self.inst.write('smub.source.output = smub.OUTPUT_OFF')
                except Exception:
                    pass
                self.inst.close()
        finally:
            self.inst = None
            self.connected = False
    # ---------------- 单通道输出 ----------------
    def output_sequence(self, y, dwell_s=0.01, compliance=0.01, on_point=None, channel="A"):
        if not self.connected:
            raise RuntimeError("Not connected")

        y = np.asarray(y, dtype=float)
        self._abort = False  # 每次开始之前复位

        if self.is_2600_tsp:
            # ---- 2600/TSP 路径（2614B 走这里）----
            ch = (channel or "A").upper()
            s = "smua" if ch == "A" else "smub"
            w, q = self.inst.write, self.inst.query

            # 基本配置
            w(f"{s}.reset()")
            w(f"{s}.source.func = {s}.OUTPUT_DCVOLTS")
            w(f"{s}.source.limiti = {float(compliance)}")
            w(f"{s}.source.output = {s}.OUTPUT_ON")
            try:
                for i, v in enumerate(y):
                    if getattr(self, "_abort", False):
                        break
                    w(f"{s}.source.levelv = {float(v)}")
                    if dwell_s > 0:
                        time.sleep(dwell_s)
                    try:
                        i_meas = float(q(f"print({s}.measure.i())"))
                    except Exception:
                        i_meas = float('nan')
                    if on_point:
                        on_point({"index": i, "V_set": float(v), "I_meas": i_meas, "V_meas": None})
            finally:
                try:
                    w(f"{s}.source.output = {s}.OUTPUT_OFF")
                finally:
                    self._abort = False  # 跑完或中断后复位
            return

        # ---- 非 2600/TSP：SCPI 路径 ----
        w, q = self.inst.write, self.inst.query
        w("*RST;*CLS")
        w(":SOUR:FUNC VOLT")
        w(f":SENS:CURR:PROT {float(compliance)}")
        w(":OUTP ON")
        try:
            for i, v in enumerate(y):
                if getattr(self, "_abort", False):   # <-- 支持停止
                    break
                w(f":SOUR:VOLT {float(v)}")
                if dwell_s > 0:
                    time.sleep(dwell_s)
                try:
                    i_meas = float(q(":MEAS:CURR?"))
                except Exception:
                    try:
                        parts = q(":READ?").strip().split(",")
                        i_meas = float(parts[1]) if len(parts) > 1 else float(parts[0])
                    except Exception:
                        i_meas = float("nan")
                if on_point:
                    on_point({"index": i, "V_set": float(v), "I_meas": i_meas, "V_meas": None})
        finally:
            try:
                w(":OUTP OFF")
            finally:
                self._abort = False

    # ---------------- 双通道同步输出（GUI Dual 模式用） ----------------
    def output_sequence_dual(self, yA, yB, dwell_s=0.01, compliance=0.01, on_point=None):
        if not self.connected:
            raise RuntimeError("Not connected")
        if not self.is_2600_tsp:
            raise RuntimeError("This instrument is single-channel in this driver; dual output not supported.")

        yA = np.asarray(yA, dtype=float)
        yB = np.asarray(yB, dtype=float)
        n = min(len(yA), len(yB))
        yA, yB = yA[:n], yB[:n]

        w, q = self.inst.write, self.inst.query
        self._abort = False  # 新一次运行复位

        for s in ("smua", "smub"):
            w(f"{s}.reset()")
            w(f"{s}.source.func = {s}.OUTPUT_DCVOLTS")
            w(f"{s}.source.limiti = {float(compliance)}")
            w(f"{s}.source.output = {s}.OUTPUT_ON")
        try:
            for i, (va, vb) in enumerate(zip(yA, yB)):
                if getattr(self, "_abort", False):
                    break
                w(f"smua.source.levelv = {float(va)}")
                w(f"smub.source.levelv = {float(vb)}")
                if dwell_s > 0:
                    time.sleep(dwell_s)
                # 只回读 A 通道电流（如需 B，可再测一次 smub.measure.i()）
                try:
                    i_meas = float(q("print(smua.measure.i())"))
                except Exception:
                    i_meas = float("nan")
                if on_point:
                    on_point({"index": i, "V_set": float(va), "I_meas": i_meas, "V_meas": None})
        finally:
            try:
                for s in ("smua", "smub"):
                    w(f"{s}.source.output = {s}.OUTPUT_OFF")
            finally:
                self._abort = False
    def safe_zero_all(self):
        """
        Abort any ongoing loop, set source level to 0 and turn outputs OFF.
        Uses TSP for 2600-series, SCPI otherwise. Safe to call repeatedly.
        """
        # request the worker loop to stop ASAP
        try:
            self._abort = True
        except Exception:
            pass

        if not self.connected or self.inst is None:
            return

        try:
            if self.is_2600_tsp:
                w = self.inst.write
                # zero level first, then outputs off (both channels)
                w('smua.source.levelv = 0')
                w('smua.source.output = smua.OUTPUT_OFF')
                w('smub.source.levelv = 0')
                w('smub.source.output = smub.OUTPUT_OFF')
            else:
                w = self.inst.write
                try:
                    w(':SOUR:VOLT 0')
                except Exception:
                    pass
                try:
                    w(':OUTP OFF')
                except Exception:
                    pass
        except Exception:
            # swallow: safety function should never raise
            pass
