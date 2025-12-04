# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 11:38:04 2025

@author: p81942ai
"""
# iv_live_window.py
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class LivePlotWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Live IV Plot")
        self.geometry("900x520")

        self.fig = Figure(figsize=(7.5, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Live: I vs V")
        self.ax.set_xlabel("Voltage (V)")
        self.ax.set_ylabel("Current (A)")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tb_frame = tk.Frame(self)
        tb_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, tb_frame).update()

