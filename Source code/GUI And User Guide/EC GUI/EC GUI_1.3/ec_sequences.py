import numpy as np
import matplotlib.pyplot as plt
from mpmath.libmp.libelefun import EXP_SERIES_U_CUTOFF


class EC_SequenceBuilder:
    def __init__(self):
        pass

    def generate_triangle_wave(self, amplitude, n_points, n_cycles=1, reverse=False):
        """
        生成闭环三角波电压序列。模式为 0→+a→-a→0，或0→-a→+a→0（reverse）。

        参数:
            amplitude: float, 振幅a
            n_points: int, 每个周期的采样点数（建议能被3整除，自动补齐）
            n_cycles: int, 总循环次数
            reverse: bool, 是否反向（三角波顺序）

        返回:
            np.ndarray, 电压序列
        """
        # 自动补齐采样点，确保完整闭环（三段：上升、下降、返回0）
        base_points = n_points // 4
        remainder = n_points % 4
        n_up = base_points + (1 if remainder > 0 else 0)
        n_down = base_points + (1 if remainder > 1 else 0)
        n_return = base_points

        if not reverse:
            # 0 → +a
            up = np.linspace(0, amplitude, n_up, endpoint=False)
            # +a → -a
            down = np.linspace(amplitude, -amplitude, n_down * 2, endpoint=False)
            # -a → 0
            ret = np.linspace(-amplitude, 0, n_return + 1)  # 最后一个点闭环
        else:
            # 0 → -a
            up = np.linspace(0, -amplitude, n_up, endpoint=False)
            # -a → +a
            down = np.linspace(-amplitude, amplitude, n_down * 2, endpoint=False)
            # +a → 0
            ret = np.linspace(amplitude, 0, n_return + 1)

        one_cycle = np.concatenate([up, down, ret])
        # 去除周期间重复点（除最后一个点闭环用）
        cycle_no_end = one_cycle[:-1]
        # 拼接周期
        seq = np.tile(cycle_no_end, n_cycles)
        # 最终闭环
        seq = np.concatenate([seq, [0]])
        return seq

    def generate_ppf_sequence(self, a, n_pulse, n_gap, n_rest=1):
        """
        构造Paired-Pulse Facilitation脉冲序列。
        :param a: float, 脉冲电压幅值
        :param n_pulse: int, 每个脉冲的采样点数
        :param n_gap: int, 两个脉冲之间的0电压点数（间隔）
        :param n_rest: int, 第二个脉冲后归零的点数（默认为1）
        :return: np.ndarray, 电压序列
        """
        pulse = np.ones(n_pulse) * a
        zero_gap = np.zeros(n_gap)
        zero_rest = np.zeros(n_rest)
        seq = np.concatenate([pulse, zero_gap, pulse, zero_rest])
        return seq

    def generate_srdp_sequence(self, V_write, n_pulse, gap_list, repeat):
        """
        构造SRDP电压序列，不同gap（频率）脉冲段首尾拼接为一个整体。
        :param V_write: float，脉冲幅值
        :param n_pulse: int，每个脉冲的采样点数
        :param gap_list: list，每种频率对应的gap点数（如[30, 15, 5, 15]）
        :param repeat: int，每种频率下脉冲个数
        :return: np.ndarray，总电压序列
        """
        pulse = np.ones(n_pulse) * V_write
        seq_total = []
        for n_gap in gap_list:
            gap = np.zeros(n_gap)
            for i in range(repeat):
                seq_total.append(pulse)
                if i != repeat:
                    seq_total.append(gap)
        seq = np.concatenate(seq_total)
        return seq

    def generate_ltp_ltd_sequence(self,
        V_read, n_read,          # read电压、每次read点数
        V_ltp, n_ltp_pulse,   # ltp脉冲幅值、宽度、次数
        V_ltd, n_ltd_pulse,   # ltd脉冲幅值、宽度、次数
        n_block1, n_block2           # 第一组、第二组重复次数
    ):
        """
        构造结构化LTP/LTD协议的电压序列。
        """
        # 单次阶段定义
        pre_read = np.ones(n_read) * V_read
        mid_read = np.ones(n_read) * V_read
        post_read = np.ones(n_read) * V_read
        ltp_pulse = np.ones(n_ltp_pulse) * V_ltp
        ltd_pulse = np.ones(n_ltd_pulse) * V_ltd

        # 一组ltp block
        ltp_block = np.concatenate([pre_read] + [ltp_pulse])
        # 一组ltd block
        ltd_block = np.concatenate([mid_read] + [ltd_pulse])

        # 按重复次数拼接
        seq = []
        for _ in range(n_block1):
            seq.append(ltp_block)
        for _ in range(n_block2):
            seq.append(ltd_block)
        seq.append(post_read)  # 最终结尾post-read

        return np.concatenate(seq)

    def generate_read_sequence(self,
        V_read,
        n_total,
        mode="const",
        n_alt=10,
    ):
        """
        生成read阶段电压序列。
        :param V_read: float, 基本read电压幅值
        :param n_total: int, 总点数
        :param mode: str, 'const'（恒定正值），'alt'（正负交替）
        :param n_alt: int, 每隔多少点切换正负（仅mode='alt'时生效）
        :return: np.ndarray
        """
        if mode == "const":
            return np.ones(n_total) * V_read
        elif mode == "alt":
            # 正负交替块数
            blocks = int(np.ceil(n_total / n_alt))
            pattern = np.array([V_read if i % 2 == 0 else -V_read for i in range(blocks)])
            seq = np.repeat(pattern, n_alt)[:n_total]
            return seq
        elif mode == "discre":
            return  np.concatenate([np.ones(n_total) * V_read, np.zeros(100)])
        else:
            raise ValueError("mode must be 'const','discrete' or 'alt'")

    def generate_retention_sequence(self,
        read_params_pre, n_read_pre,
        read_params_post, n_read_post,
        V_write, n_pulse, n_gap, cycle
    ):
        """
        构造stp-to-ltp完整协议序列。
        """
        read_unit_pre = self.generate_read_sequence(**read_params_pre)
        read_unit_post = self.generate_read_sequence(**read_params_post)
        read_pre = np.tile(read_unit_pre, n_read_pre)
        read_post = np.tile(read_unit_post, n_read_post)
        pulse = np.ones(n_pulse) * V_write
        write_seq = []
        for i in range(cycle):
            write_seq.append(pulse)
            if i != cycle - 1:
                write_seq.append(np.zeros(n_gap))
        write_seq = np.concatenate(write_seq)
        seq = np.concatenate([read_pre, write_seq, read_post])
        return seq

    def generate_square_pulse(self,V, length, space):
        """生成方波写脉冲"""
        return np.concatenate([np.ones(length) * V, np.zeros(space)])

    def generate_triangle_pulse(self,V, step, space):
        """生成三角波写脉冲，范围 0→V_max→0，长度2*step"""
        up = np.linspace(0, V, step, endpoint=False)
        down = np.linspace(V, 0, step)
        return np.concatenate([up, down, np.zeros(space)])

    def build_stdp_sequence(self,
        read_seq,      # 你的read序列，np.ndarray
        pre_seq,     # 写脉冲shape
        post_seq,     # reset脉冲shape
        delay_points,  # pre-post间隔
        read_len,      # read区段长度
        mode
    ):
        if mode == "bef":
            # 通道A
            v_seq_a = np.concatenate([
                read_seq,
                pre_seq,
                np.zeros(delay_points),
                read_seq
            ])
        # 通道B
            v_seq_b = np.concatenate([
                np.zeros(read_len),
                np.zeros(delay_points),
                post_seq,
                np.zeros(read_len)
            ])
        else:
            v_seq_a = np.concatenate([
                read_seq,
                np.zeros(delay_points),
                pre_seq,
                read_seq
            ])
            # 通道B
            v_seq_b = np.concatenate([
                np.zeros(read_len),
                post_seq,
                np.zeros(delay_points),
                np.zeros(read_len)
            ])
        return v_seq_a, v_seq_b

    def generate_tstdp_sequence(self, read_seq, pre_seq, post_seq, mode, t_1, t_2):
        read_seq = read_seq
        pre_seq = pre_seq
        post_seq = post_seq

        if mode == "pre-post-pre":
            suma_writing = np.concatenate([ pre_seq,
                                    np.zeros(abs(abs(t_2) + abs(t_1) - len(pre_seq))),
                                    pre_seq])
            sumb_writing = np.concatenate([ np.zeros(abs(t_1)),
                                    post_seq])
        elif mode == "pre-pre-post":
            suma_writing = np.concatenate([ pre_seq,
                                    np.zeros(abs(abs(t_1) - abs(t_2) - len(pre_seq))),
                                    pre_seq,
                                    np.zeros(abs(t_2))])
            sumb_writing = np.concatenate([ np.zeros(abs(t_1)),
                                    post_seq])
        elif mode == "post-pre-pre":
            suma_writing = np.concatenate([np.zeros(abs(t_1)),
                                    pre_seq,
                                    np.zeros(abs(abs(t_2) - abs(t_1) - len(pre_seq))),
                                    pre_seq])
            sumb_writing = np.concatenate([
                                    post_seq])
        elif mode == "post-pre-post":
            suma_writing = np.concatenate([ np.zeros(abs(t_1)),
                                    pre_seq])
            sumb_writing = np.concatenate([post_seq,
                                    np.zeros(abs(abs(t_2) + abs(t_1) - len(pre_seq))),
                                    post_seq])
        elif mode == "post-post-pre":
            suma_writing = np.concatenate([np.zeros(abs(t_1)),
                                    pre_seq])
            sumb_writing = np.concatenate([post_seq,
                                    np.zeros(abs(abs(t_1) - abs(t_2) - len(pre_seq))),
                                    post_seq,
                                    np.zeros(abs(t_2))])
        elif mode == "pre-post-post":
            suma_writing = np.concatenate([pre_seq])
            sumb_writing = np.concatenate([np.zeros(abs(t_1)),
                                    post_seq,
                                    np.zeros(abs(abs(t_2) - abs(t_1) - len(pre_seq))),
                                    post_seq])
        else:
            raise ValueError("Invalid mode")

        max_len = max(len(suma_writing), len(sumb_writing))
        suma_writing = np.concatenate([suma_writing, np.zeros(max_len - len(suma_writing))])
        sumb_writing = np.concatenate([sumb_writing, np.zeros(max_len - len(sumb_writing))])
        suma = np.concatenate([read_seq,suma_writing,read_seq])
        sumb = np.concatenate([np.zeros(len(read_seq)), sumb_writing, np.zeros(len(read_seq))])
        return suma, sumb
    def generate_sine_wave(
        self,
        amplitude,
        n_points=None,           # points per cycle (use this OR freq_hz+dwell_s)
        n_cycles=1,
        start_at="zero_rise",    # 'pos_max'|'neg_max'|'zero_rise'|'zero_fall'|float radians|'90deg'
        end_at=None,             # same accepted forms as start_at, or None
        freq_hz=None,            # alternatively select frequency...
        dwell_s=None,            # ...and sample dwell (seconds) to deduce points per cycle
        dc_offset=0.0,
        include_endpoint=False,  # add the terminal sample at the end anchor (if grid-aligned)
        min_ppc=4                # safety: at least this many samples per cycle
    ):
        """
        Generate a sine wave sequence with configurable start/end anchors or explicit phase.

        You may specify either:
          - n_points (samples per cycle), OR
          - freq_hz together with dwell_s (so points-per-cycle ~= 1/(f*dwell))

        Anchors:
          'pos_max'   -> +pi/2
          'neg_max'   -> 3*pi/2
          'zero_rise' -> 0
          'zero_fall' -> pi
          numeric (float) -> phase in radians
          '<deg>deg' -> phase in degrees, e.g. '90deg'
        """
        import numpy as _np
        _pi = _np.pi

        def _anchor_to_phase(a):
            if a is None:
                return None
            if isinstance(a, (int, float)):
                return float(a)
            s = str(a).strip().lower()
            if s.endswith("deg"):
                return float(s[:-3]) * _pi / 180.0
            if s in ("pos_max", "positive_max", "max", "+max", "pmax"):
                return 0.5 * _pi
            if s in ("neg_max", "negative_max", "-max", "nmax"):
                return 1.5 * _pi
            if s in ("zero_rise", "0+", "rise", "zero_up"):
                return 0.0
            if s in ("zero_fall", "0-", "fall", "zero_down"):
                return _pi
            # fallback: try parse numeric
            try:
                return float(s)
            except Exception:
                raise ValueError(f"Unrecognized anchor/phase: {a}")

        # Work out samples-per-cycle (ppc)
        if freq_hz is not None:
            if dwell_s is None:
                raise ValueError("dwell_s must be provided when freq_hz is used.")
            ppc = max(int(round(1.0 / (float(freq_hz) * float(dwell_s)))), int(min_ppc))
        else:
            if n_points is None:
                raise ValueError("Provide either n_points per cycle, or freq_hz+dwell_s.")
            ppc = int(n_points)
            if ppc < min_ppc:
                ppc = min_ppc

        phi_start = _anchor_to_phase(start_at)
        phi_end   = _anchor_to_phase(end_at)

        step = 2.0 * _pi / ppc
        base_steps = int(round(ppc * n_cycles))

        # If an end anchor is requested, add the needed extra steps
        extra = 0
        if phi_end is not None:
            dphi = (phi_end - phi_start) % (2.0 * _pi)
            extra = int(round(dphi / step))

        # total number of samples; optionally include the exact end point
        total = base_steps + extra + (1 if include_endpoint else 0)

        # Build the phase ramp and signal
        k = _np.arange(total, dtype=float)
        theta = phi_start + k * step
        y = dc_offset + amplitude * _np.sin(theta)
        return y

class TimeSeriesPrediction:
    def __init__(self):
        pass

    def lorenz(self, length=2000, dt=0.01, sigma=10.0, beta=8 / 3, rho=28.0, seed=None):
        # np.random.seed(seed)
        x, y, z = 1.0, 1.0, 1.0
        xs = []
        for i in range(length):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x += dx
            y += dy
            z += dz
            xs.append(x)
        return np.array(xs)

    def mackey_glass(self, length=1000, tau=17, beta=0.2, gamma=0.1, n=10, seed=42):
        np.random.seed(seed)
        x = np.zeros(length + tau + 1)
        x[0] = 1.2
        for t in range(1, length + tau):
            x_tau = x[t - tau] if t >= tau else 0.0
            x[t] = x[t - 1] + (beta * x_tau / (1 + x_tau ** n) - gamma * x[t - 1])
        return x[tau:]

    def narma10(self, length=2000, seed=42):
        np.random.seed(seed)
        u = np.random.uniform(0, 0.5, length)  # 更常见设置
        y = np.zeros(length)
        y[:10] = 0.1  # 初始化前10步，避免全0
        for t in range(10, length):
            y[t] = 0.3 * y[t - 1] \
                   + 0.05 * y[t - 1] * np.sum(y[t - 10:t]) \
                   + 1.5 * u[t - 10] * u[t - 1] \
                   + 0.1
            # 裁剪防止爆炸
            if abs(y[t]) > 10:
                y[t] = 10 * np.sign(y[t])
        return y

    def henon(self, length=2000, a=1.4, b=0.3, x0=0.1, y0=0.1, seed=None):
        x, y = x0, y0
        xs = []
        for _ in range(length):
            x, y = 1 - a * x ** 2 + y, b * x
            xs.append(x)
        return np.array(xs)


# builder = TimeSeriesPrediction() #EC_SequenceBuilder()
# vseq = builder.lorenz(length=4000) #narma10, mackey_glass, lorenz, henon
builder = EC_SequenceBuilder()
# STP-to-LTP or Natural decay
read_params_pre = dict(V_read=0.1, n_total=40, mode="const", n_alt=10)   # mode可以是"const", "alt", "discre"
read_params_post = dict(V_read=0.1, n_total=40, mode="discre", n_alt=10)
vseq = builder.generate_retention_sequence(
    read_params_pre=read_params_pre, n_read_pre=5,
    read_params_post=read_params_post, n_read_post=2,
    V_write=1, n_pulse=5, n_gap=2, cycle=10
)


# 绘图
plt.figure(figsize=(8, 4))
plt.plot(vseq[0], marker='.', linewidth=1, label = "pre")
plt.plot(vseq[1], marker='o', linewidth=1, label = "post")
plt.xlabel("Sample Index")
plt.ylabel("Voltage (V)")
plt.title("Generated Triangle Wave Voltage Sequence")
plt.grid(True)
plt.tight_layout()
plt.show()

'''
vseq = builder.generate_triangle_wave(
    amplitude=1.0,
    n_points=100,
    n_cycles=1,
    reverse=False)


vseq = generate_ppf_sequence(
    a=0.8, 
    n_pulse=10, 
    n_gap=1, 
    n_rest=0) # n_rest只有在需要连续测试的时候才会启用，即自动化完整ppf序列测试，且需要先找到器件完全恢复到初始状态的时间

vseq = generate_srdp_sequence(
    V_write=0.8, 
    n_pulse=5, 
    gap_list=[30, 10, 1, 10], 
    repeat=10)

vseq = generate_ltp_ltd_sequence(
    V_read=0.1, n_read=5,
    V_ltp=0.8, n_ltp_pulse=5,
    V_ltd=-1.0, n_ltd_pulse=5,
    n_block1=10, n_block2=10
)

# STP-to-LTP or Natural decay
read_params_pre = builder.dict(V_read=0.1, n_total=40, mode="const", n_alt=10)   # mode可以是"const", "alt", "discre"
read_params_post = builder.dict(V_read=0.1, n_total=40, mode="discre", n_alt=10)
vseq = builder.generate_retention_sequence(
    read_params_pre=read_params_pre, n_read_pre=5,
    read_params_post=read_params_post, n_read_post=2,
    V_write=1, n_pulse=5, n_gap=2, cycle=10
)


read_seq = builder.generate_read_sequence(V_read=0.1, n_total=10, mode="alt", n_alt=10)
spike_pre = builder.generate_triangle_pulse(V=1, step=10, space=0)
reset_pre = builder.generate_triangle_pulse(V=-1, step=10, space=0)
spike_post = builder.generate_triangle_pulse(V=1, step=10, space=0)
reset_post = builder.generate_triangle_pulse(V=-1, step=10, space=0)
pre_seq = builder.np.concatenate([spike_pre, reset_post])
post_seq = builder.np.concatenate([spike_post, reset_post])
delay_points = builder.np.linspace(1, len(pre_seq), 6)

vseq = builder.build_stdp_sequence(
    read_seq = read_seq,      # 你的read序列，np.ndarray
    pre_seq = pre_seq,     # 写脉冲shape
    post_seq = post_seq,     # reset脉冲shape
    delay_points = 100,  # pre-post间隔
    read_len=len(read_seq),      # read区段长度
    mode="bef",
)


    t_prepostpre = [
        [0.6, -0.6],[0.8, -0.6],[1.0, -0.6],[1.1, -0.6],[1.2, -0.6],[1.3, -0.6],[1.4, -0.6],
        [0.6, -0.8],[0.8, -0.8],[1.0, -0.8],[1.1, -0.8],[1.2, -0.8],[1.3, -0.8],[1.4, -0.8],
        [0.6, -1.0],[0.8, -1.0],[1.0, -1.0],[1.1, -1.0],[1.2, -1.0],[1.3, -1.0],[1.4, -1.0],
        [0.6, -1.1],[0.8, -1.1],[1.0, -1.1],[1.1, -1.1],[1.2, -1.1],[1.3, -1.1],[1.4, -1.1],
        [0.6, -1.2],[0.8, -1.2],[1.0, -1.2],[1.1, -1.2],[1.2, -1.2],[1.3, -1.2],[1.4, -1.2],
        [0.6, -1.3],[0.8, -1.3],[1.0, -1.3],[1.1, -1.3],[1.2, -1.3],[1.3, -1.3],[1.4, -1.3],
        [0.6, -1.4],[0.8, -1.4],[1.0, -1.4],[1.1, -1.4],[1.2, -1.4],[1.3, -1.4],[1.4, -1.4],]
    t_preprepost = [
        [1.1, 0.1], [1.2, 0.2], [1.3, 0.3], [1.4, 0.4],
        [1.2, 0.1], [1.3, 0.2], [1.4, 0.3],
        [1.3, 0.1], [1.4, 0.2],
        [1.4, 0.1], ]
    t_postprepre = [
        [-0.1, -1.1], [-0.2, -1.2], [-0.3, -1.3], [-0.4, -1.4],
        [-0.1, -1.2], [-0.1, -1.3], [-0.1, -1.4],
        [-0.2, -1.3], [-0.2, -1.4],
        [-0.3, -1.4], ]

read_seq = generate_read_sequence(V_read=0.1, n_total=10, mode="alt", n_alt=10)
spike_pre = generate_triangle_pulse(V=1, step=10, space=0)
reset_pre = generate_triangle_pulse(V=-1, step=10, space=0)
spike_post = generate_triangle_pulse(V=1, step=10, space=0)
reset_post = generate_triangle_pulse(V=-1, step=10, space=0)
pre_seq = np.concatenate([spike_pre, reset_post])
post_seq = np.concatenate([spike_post, reset_post])

vseq = generate_tstdp_sequence(read_seq=read_seq, pre_seq=pre_seq, post_seq=post_seq, mode="post-pre-post", t_1=-20, t_2=100)

'''