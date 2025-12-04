

# %% [1] Introduction

# %% [2] Libraries that imported
# ======================================= Libraries ==========================================
import os  # file path operations
import re  # regular expression matching (e.g. concentration)
import random  # generating random colors
import tkinter as tk  # main Tkinter module
from tkinter import filedialog, messagebox, ttk # file dialog, message boxes, combobox
import pandas as pd  # data loading and table operations
import numpy as np  # numerical computing
import matplotlib.pyplot as plt  # plotting library
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # embedding matplotlib in Tkinter
from scipy.stats import linregress  # linear regression fitting
import mplcursors  # For interactive data cursor (hover to display values)
import webbrowser
from PIL import Image, ImageTk
from io import BytesIO

# %% [3] General constants
# ======================================= Constants ==========================================
# Define constant unit used in this code
unit_factors = {
    "V": 1, "mV": 1e-3, "μV": 1e-6,
    "A": 1, "mA": 1e-3, "μA": 1e-6, "nA": 1e-9, "pA": 1e-12,
    "S": 1, "mS": 1e-3, "μS": 1e-6, "nS": 1e-9,
    "S/m": 1, "mS/m": 1e-3, "μS/m": 1e-6,
    "M": 1, "mM": 1e-3, "μM": 1e-6
}

e = 1.602e-19       # Elementary charge (Coulombs)
k_B = 1.381e-23     # Boltzmann constant (Joule/Kelvin)
T = 300             # Temperature in Kelvin (room temperature)
N_A = 6.022e23      # Avogadro's number (1/mol)
F = 96485.33      # Faraday constant (Coulombs per mole of electrons)
R = 8.314    # Ideal gas constant (Joules per mole per Kelvin)
# %% [4] Global State and Plot Configuration Settings
# ================================= Global State and Plot Configuration ============================
figures = []  # Store matplotlib figure objects
conductance_results = []  # Store experimental conductance results
theoretical_results = []  # Store theoretical conductance results
drift_diffusion_results = [] # Stores results from drift-diffusion calculations: includes total potential, redox potential, Em, mobilities, etc.
drift_diffusion_table = None  # Will be assigned the Treeview widget instance used for the drift-diffusion result table
ion_input_entries = {} # Dictionary mapping ion input label names (e.g., 'γ_H', 'C_L') to their corresponding Entry widgets for easy access
conductance_table = {}  # key: filename, value: (concentration, conductance)

# Dictionary to store geometric parameters of the nanochannel (in meters)
nanochannel_dimensions = {
    "length": None,   # Channel length (m)
    "width": None,    # Channel width (m)
    "height": None,   # Channel height (m)
    "number": None    # Number of nanochannels
}

# === Global Matplotlib Font Settings ===
plt.rcParams.update({
    "font.family": "sans-serif",   # Font family (e.g., "Times New Roman" or "Arial")
    "font.size": 14,               # Global default font size
    "axes.titlesize": 16,          # Title font size for individual plots
    "axes.labelsize": 14,          # Font size for axis labels
    "xtick.labelsize": 12,         # Font size for x-axis tick labels
    "ytick.labelsize": 12,         # Font size for y-axis tick labels
    "legend.fontsize": 12,         # Font size for legend text
    "figure.titlesize": 16         # Font size for the overall figure title
})

# %% [5] General Utility Functions
# These helper functions are frequently called throughout the program.
# Includes: plotting utilities, unit conversion, zoom controls.

# Add general buttion function (save data, save plot, close plot)
def add_plot_utilities(fig, ax, plot_frame, x, y, x_label="X", y_label="Y", filename_prefix="plot"):
    """Add interactive utility buttons for saving plot, saving data, and closing the plot frame."""

    def save_plot():  # Save current figure to PNG
        file_path = filedialog.asksaveasfilename(  # Open file dialog for saving image
            defaultextension=".png",  # Default file extension
            filetypes=[("PNG Image", "*.png")],  # Allowed file types
            title="Save Plot Image"  # Dialog title
        )
        if file_path:  # If user selected a path
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')  # Save figure with high resolution
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save plot:\n{e}")  # Show error if saving fails

    def save_data():  # Save x/y data to CSV
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv")],
            title="Save Plotting Data"
        )
        if file_path:  # If user selected a path
            try:
                df = pd.DataFrame({x_label: x, y_label: y})  # Create DataFrame from x and y data
                df.to_csv(file_path, index=False)  # Export to CSV without index column
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save data:\n{e}")  # Show error if saving fails

    def close_plot():  # Close and destroy the plot frame from GUI
        plot_frame.destroy()

    fig.savefig(f"{filename_prefix}.png", dpi=150, bbox_inches='tight')  # Automatically save a default version

    btn_frame = tk.Frame(plot_frame)  # Create a container for the buttons
    btn_frame.pack(pady=4)  # Add spacing below button frame

    tk.Button(btn_frame, text="Save Plot", command=save_plot).pack(side=tk.LEFT, padx=5)  # Save image button
    tk.Button(btn_frame, text="Save Data", command=save_data).pack(side=tk.LEFT, padx=5)  # Save data button
    tk.Button(btn_frame, text="Close This Plot", command=close_plot).pack(side=tk.LEFT, padx=5)  # Close plot button

# Select unit to allocate for different functions
def create_unit_selector(parent, label_text, row, column, options, default_value):
    """Create a labeled unit selection dropdown (ttk.Combobox) and place it in a grid layout."""
    label = tk.Label(parent, text=label_text)  # Create a label with the provided text
    label.grid(row=row, column=column, sticky="e")  # Place the label in the grid (right aligned)
    selector = ttk.Combobox(parent, values=options, width=6, state="readonly")  # Create a readonly dropdown with unit options
    selector.set(default_value)  # Set default selected unit
    selector.grid(row=row, column=column + 1)  # Place the dropdown next to the label
    return selector  # Return the dropdown widget for later access

# Convert unit for rescale
def apply_unit_conversion(x_data, y_data, x_from, x_to, y_from, y_to):
    """
    Apply unit conversion to x and y data based on the original and target units.
    This function rescales numerical data according to the physical principle:
        value_new = value_old × (unit_old / unit_new)
    For example, converting current from μA to A means multiplying by 1e-6.
    
    Parameters:
        x_data (array): original x data (e.g., voltage or concentration)
        y_data (array): original y data (e.g., current or conductance)
        x_from (str): original unit of x (e.g., "V", "mV")
        x_to (str): target unit of x
        y_from (str): original unit of y (e.g., "nA", "μS")
        y_to (str): target unit of y

    Returns:
        tuple: (converted x data, converted y data)
    """
    try:
        x_factor = unit_factors[x_from] / unit_factors[x_to]  # Compute conversion factor for x: (unit_old / unit_new)
        y_factor = unit_factors[y_from] / unit_factors[y_to]  # Compute conversion factor for y: (unit_old / unit_new)
        return x_data * x_factor, y_data * y_factor  # Multiply original values by conversion factors
    except KeyError as e:
        raise ValueError(f"Unsupported unit conversion: {e}")  # Raise error if units are not recognized

def attach_zoom_controls(parent_frame, fig, ax):
    """
    Attach zoom control buttons (Zoom In/Out, Set Center, Reset) to the given parent frame.
    Works on the provided matplotlib figure and axis, without any pop-up windows.
    """
    zoom_center = {'x': 0, 'y': 0}
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    def zoom(factor):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_center = zoom_center['x']
        y_center = zoom_center['y']
        x_range = (xlim[1] - xlim[0]) / 2 / factor
        y_range = (ylim[1] - ylim[0]) / 2 / factor
        ax.set_xlim(x_center - x_range, x_center + x_range)
        ax.set_ylim(y_center - y_range, y_center + y_range)
        fig.canvas.draw()

    def reset_zoom():
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)
        fig.canvas.draw()

    # keep a reference so we can disconnect
    cid_holder = {'cid': None}

    def set_zoom_center():
        # visual cue
        tk_widget = fig.canvas.get_tk_widget()
        old_cursor = tk_widget['cursor']
        tk_widget.configure(cursor='crosshair')
        set_btn.config(text="Click on plot to set center…")

        def onclick(event):
            if event.inaxes == ax:
                zoom_center['x'] = event.xdata
                zoom_center['y'] = event.ydata
                # clean up
                fig.canvas.mpl_disconnect(cid_holder['cid'])
                tk_widget.configure(cursor=old_cursor)
                set_btn.config(text="Set Zoom Center")

        # connect and store id
        cid_holder['cid'] = fig.canvas.mpl_connect('button_press_event', onclick)

    # Buttons
    set_btn = tk.Button(parent_frame, text="Set Zoom Center", command=set_zoom_center)
    set_btn.pack(side=tk.LEFT, padx=4)

    tk.Button(parent_frame, text="Zoom In", command=lambda: zoom(1.5)).pack(side=tk.LEFT, padx=4)
    tk.Button(parent_frame, text="Zoom Out", command=lambda: zoom(1/1.5)).pack(side=tk.LEFT, padx=4)
    tk.Button(parent_frame, text="Reset Zoom", command=reset_zoom).pack(side=tk.LEFT, padx=4)

# Provides axis rescaling with unit converter function
def attach_unit_conversion_ui(parent_frame, func):
    """
    Left: unit selectors + Rescale
    Right: (placeholder) an empty frame to host zoom controls later.
    """
    # Destroy any existing unit conversion UI if present
    if hasattr(func, "unit_frame") and func.unit_frame.winfo_exists():
        func.unit_frame.destroy()

    # Container for the entire control row
    func.unit_frame = tk.Frame(parent_frame)
    func.unit_frame.pack(pady=5, fill=tk.X)

    main_row = tk.Frame(func.unit_frame)
    main_row.pack(fill=tk.X, expand=True)

    # Left block: unit selectors + Rescale
    left_block = tk.Frame(main_row)
    left_block.pack(side=tk.LEFT, anchor='w', padx=5)

    func.v_unit_orig = create_unit_selector(left_block, "Original Voltage Unit:", 0, 0, list(unit_factors.keys()), "V")
    func.v_unit_target = create_unit_selector(left_block, "Target Voltage Unit:", 0, 2, list(unit_factors.keys()), "V")
    func.i_unit_orig = create_unit_selector(left_block, "Original Current Unit:", 1, 0, list(unit_factors.keys()), "A")
    func.i_unit_target = create_unit_selector(left_block, "Target Current Unit:", 1, 2, list(unit_factors.keys()), "A")

    def rescale():
        if hasattr(func, "plot_frame") and func.plot_frame.winfo_exists():
            func.plot_frame.destroy()
        func(rescale=True)

    tk.Button(left_block, text="Rescale", command=rescale).grid(row=2, column=0, columnspan=4, pady=6)

    # Right block: placeholder for zoom controls (no buttons here)
    right_block = tk.Frame(main_row)
    right_block.pack(side=tk.RIGHT, anchor='e', padx=10)

    tk.Label(right_block, text="Zoom Controls:").pack(side=tk.LEFT, padx=(0, 5))

    # Expose this frame so caller can attach zoom controls once fig/ax exist
    func.zoom_block_frame = right_block

# %% [6] Main Function 1 : Nanochannel Details
def show_nanochannel_inputs():
    import csv
    from tkinter import filedialog, messagebox, ttk

    global entry_device_name, entry_length, entry_width, entry_height, entry_number
    global device_combobox

    # 清空右侧界面
    for widget in right_frame.winfo_children():
        widget.destroy()

    
    
    # 通用创建 entry 函数
    def create_labeled_entry(parent, label_text, default_val, var_name):
        row = tk.Frame(parent)
        row.pack(anchor="w", pady=2)
        tk.Label(row, text=label_text, width=25).pack(side=tk.LEFT)
        entry = tk.Entry(row, width=20)
        entry.insert(0, default_val)
        entry.pack(side=tk.LEFT)
        globals()[var_name] = entry
        return entry

    

    # 存储维度到全局变量
    def confirm_dimensions():
        try:
            nanochannel_dimensions["length"] = float(entry_length.get())
            nanochannel_dimensions["width"] = float(entry_width.get())
            nanochannel_dimensions["height"] = float(entry_height.get())
            nanochannel_dimensions["number"] = float(entry_number.get())
            messagebox.showinfo("Success", "Nanochannel dimensions updated and stored.")
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    

    # 保存当前输入到 CSV 和列表
    def add_dimensions_to_list():
        device = entry_device_name.get().strip()
        if not device:
            messagebox.showwarning("Missing Name", "Please enter a device name.")
            return
        try:
            data = {
                "device": device,
                "length": float(entry_length.get()),
                "width": float(entry_width.get()),
                "height": float(entry_height.get()),
                "number": float(entry_number.get())
            }

            file_path = "stored_devices.csv"
            file_exists = os.path.isfile(file_path)
            with open(file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["device", "length", "width", "height", "number"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)

            # 更新下拉框
            current = device_combobox["values"]
            if device not in current:
                device_combobox["values"] = (*current, device)
            messagebox.showinfo("Saved", f"Device '{device}' added.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add dimensions:\n{e}")



    def load_selected_device():
        device = device_combobox.get()
        if not device:
            messagebox.showwarning("No Selection", "Please select a device.")
            return
        try:
            with open("stored_devices.csv", "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["device"] == device:
                        entry_device_name.delete(0, tk.END)
                        entry_length.delete(0, tk.END)
                        entry_width.delete(0, tk.END)
                        entry_height.delete(0, tk.END)
                        entry_number.delete(0, tk.END)

                        entry_device_name.insert(0, row["device"])
                        entry_length.insert(0, row["length"])
                        entry_width.insert(0, row["width"])
                        entry_height.insert(0, row["height"])
                        entry_number.insert(0, row["number"])
                        messagebox.showinfo("Loaded", f"Dimensions for '{device}' loaded.")
                        return
                messagebox.showwarning("Not Found", f"Device '{device}' not found in file.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load dimensions:\n{e}")


    # 初始化 device_combobox 的内容
    def initialize_device_list():
        try:
            if os.path.exists("stored_devices.csv"):
                with open("stored_devices.csv", "r", newline="") as f:
                    reader = csv.DictReader(f)
                    devices = sorted({row["device"] for row in reader})
                    device_combobox["values"] = devices
        except Exception as e:
            print(f"Warning: failed to initialize device list: {e}")

    

    
    def save_dimensions_to_file():
        try:
            device = entry_device_name.get().strip()
            length = float(entry_length.get())
            width = float(entry_width.get())
            height = float(entry_height.get())
            number = float(entry_number.get())

            data = {
                "device": device,
                "length": length,
                "width": width,
                "height": height,
                "number": number
            }

            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Nanochannel Dimensions"
            )

            if file_path:
                file_exists = os.path.isfile(file_path)
                with open(file_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["device", "length", "width", "height", "number"])
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(data)
                messagebox.showinfo("Saved", f"Dimensions saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save dimensions:\n{e}")

    # === 统一按钮区域排布（放在函数最后） ===
    frame = tk.Frame(right_frame)
    frame.pack(pady=20)
    # Select Device + Load
    list_row = tk.Frame(frame)
    list_row.pack(pady=5)
    tk.Label(list_row, text="Select Device:", width=15).pack(side=tk.LEFT)
    device_combobox = ttk.Combobox(list_row, width=20, state="readonly")
    device_combobox.pack(side=tk.LEFT, padx=5)
    tk.Button(list_row, text="Load Dimensions", command=load_selected_device, width=15).pack(side=tk.LEFT, padx=5)
    tk.Button(list_row, text="Add Dimensions to List", command=add_dimensions_to_list, width=20).pack(side=tk.LEFT, padx=5)

    initialize_device_list()
    # 添加 Device Name 输入框
    create_labeled_entry(frame, "Device Name:", "", "entry_device_name")

    # 添加维度输入框
    create_labeled_entry(frame, "Channel length (m):", "", "entry_length")
    create_labeled_entry(frame, "Channel width (m):", "", "entry_width")
    create_labeled_entry(frame, "Channel height (m):", "", "entry_height")
    create_labeled_entry(frame, "Channel number:", "", "entry_number")

    # Save to File
    tk.Button(frame, text="Save Dimensions to File", command=save_dimensions_to_file, width=25).pack(pady=5)

    # Confirm Done
    tk.Button(frame, text="Done", command=confirm_dimensions, width=20).pack(pady=10)

# %% [7] Main Function 2 : Electrolyte Anylysis + Theoretical G; Combine Exp/Theo, Compare G & σ 
def open_theoretical_conductance_popup(results):

    def save_table_to_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not file_path:
            return
        df_out = pd.DataFrame([tree.item(row)['values'] for row in tree.get_children()],
                              columns=["Concentration (M)", "Theoretical Conductance (S)"])
        df_out.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Table saved to:\n{file_path}")

    def delete_selected_row():
        selected = tree.selection()
        for item in selected:
            tree.delete(item)

    def on_double_click(event):
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if not item or not column:
            return
        # col = int(column.replace('#', '')) - 1
        x, y, width, height = tree.bbox(item, column)
        value = tree.set(item, column)

        entry = tk.Entry(tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus()

        def on_enter(event):
            tree.set(item, column, entry.get())
            entry.destroy()

        entry.bind('<Return>', on_enter)
        entry.bind('<FocusOut>', lambda e: entry.destroy())

    popup = tk.Toplevel()
    popup.title("Theoretical Conductance Table")

    cols = ["Concentration (M)", "Theoretical Conductance (S)"]
    tree = ttk.Treeview(popup, columns=cols, show="headings")
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=200)
    for conc, g in results:
        tree.insert("", "end", values=[conc, f"{g:.4e}"])
    tree.bind("<Double-1>", on_double_click)
    tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    btn_frame = tk.Frame(popup)
    btn_frame.pack(pady=5)

    tk.Button(btn_frame, text="Save Table", command=save_table_to_csv).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Delete Selected Row", command=delete_selected_row).pack(side=tk.LEFT, padx=5)

def enter_electrolyte_details(right_frame):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import pandas as pd

    global user_selection_tree_data
    user_selection_tree_data = []

    for widget in right_frame.winfo_children():
        widget.destroy()

    # ==================== Top Frame for Unit Conversion =====================
    top_frame = tk.Frame(right_frame)
    top_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

    tk.Label(top_frame, text="Convert Value").grid(row=0, column=0, columnspan=2)
    tk.Label(top_frame, text="Value:").grid(row=1, column=0)
    value_entry = tk.Entry(top_frame, width=14)
    value_entry.grid(row=1, column=1)
    tk.Label(top_frame, text="Concentration (mol/L):").grid(row=2, column=0)
    conc_entry = tk.Entry(top_frame, width=14)
    conc_entry.grid(row=2, column=1)
    tk.Label(top_frame, text="From Unit:").grid(row=3, column=0)
    unit_options = ["S/m", "mS/cm", "μS/cm", "mS/m", "μS/m", "S·cm²/mol", "S·m²/mol"]
    from_unit = ttk.Combobox(top_frame, values=unit_options, width=12)
    from_unit.grid(row=3, column=1)
    tk.Label(top_frame, text="To Unit:").grid(row=4, column=0)
    to_unit = ttk.Combobox(top_frame, values=unit_options, width=12)
    to_unit.grid(row=4, column=1)
    result_label = tk.Entry(top_frame, width=14)
    result_label.grid(row=5, column=1)
    tk.Button(top_frame, text="Transfer", command=lambda: convert_units(value_entry, conc_entry, from_unit, to_unit, result_label), width=14).grid(row=5, column=0)

    # ==================== Electrolyte Selection Dropdown =====================
    select_frame = tk.Frame(right_frame)
    select_frame.pack(pady=5)
    
    tk.Label(select_frame, text="Select Salt:").pack(side=tk.LEFT)
    
    # NOTE: give StringVar an explicit master
    salt_var = tk.StringVar(master=right_frame)
    
    salt_keys = ["KCl", "LiCl", "Other"]
    salt_cb = ttk.Combobox(select_frame,
                       textvariable=salt_var,
                       values=salt_keys,
                       width=20,
                       state="readonly")
    salt_cb.pack(side=tk.LEFT, padx=5)

    # ==================== Electrolyte Table =====================
    table_frame = tk.Frame(right_frame)
    table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    columns = ("Concentration_M", "Theoretical Conductivity_S_per_m")
    user_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
    for col in columns:
        user_tree.heading(col, text=col)
        user_tree.column(col, anchor="center")
    user_tree.pack(fill=tk.BOTH, expand=True)

    # ==================== Button Functions =====================
    def add_blank_row_user():
        user_tree.insert("", "end", values=("", ""))

    def save_user_table():
        global user_selection_tree_data
        user_selection_tree_data = [user_tree.item(row)['values'] for row in user_tree.get_children()]
        messagebox.showinfo("Saved", "Your conductivity data is saved and ready for theoretical calculation.")

    def save_user_table_to_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not file_path:
            return
        df_out = pd.DataFrame([user_tree.item(row)['values'] for row in user_tree.get_children()], columns=columns)
        df_out.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Table saved to:\n{file_path}")

    def clear_user_table():
        for row in user_tree.get_children():
            user_tree.delete(row)

    def delete_selected_user_row():
        selected = user_tree.selection()
        for item in selected:
            user_tree.delete(item)

    def fill_from_selection(event=None):
        try:
            user_tree.delete(*user_tree.get_children())
            key = (salt_var.get() or "").strip()
            
            if key in salt_data_dict:
                def to_float(s):
                    try:
                        return float(str(s).replace("E", "e"))
                    except Exception:
                        return float("-inf")
                rows = sorted(salt_data_dict[key], key=lambda x: to_float(x[0]), reverse=True)
                for conc, cond in rows:
                    user_tree.insert("", "end", values=[conc, cond])
            else:
                user_tree.insert("", "end", values=("", ""))

            # TEMP debug: see if handler runs and how many rows inserted
            # print("fill_from_selection:", key, "rows:", len(user_tree.get_children()))
        except Exception as e:
            messagebox.showerror("Fill error", f"{e}")



    # 1) mouse selection
    salt_cb.bind("<<ComboboxSelected>>", fill_from_selection)
    # 2) variable change (backstop in case event doesn’t fire on some setups)
    salt_var.trace_add('write', lambda *_: fill_from_selection())
    # 3) default select and initial fill so the table isn’t empty on open
    
    


    # ==================== Double-click Edit =====================
    def on_double_click(event):
        item = user_tree.identify_row(event.y)
        column = user_tree.identify_column(event.x)
        if not item or not column:
            return
        #col = int(column.replace('#', '')) - 1
        x, y, width, height = user_tree.bbox(item, column)
        value = user_tree.set(item, column)

        entry = tk.Entry(user_tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus()

        def on_enter(event):
            user_tree.set(item, column, entry.get())
            entry.destroy()

        entry.bind('<Return>', on_enter)
        entry.bind('<FocusOut>', lambda e: entry.destroy())

    user_tree.bind("<Double-1>", on_double_click)

    # ==================== Bottom Buttons =====================
    bottom_frame = tk.Frame(right_frame)
    bottom_frame.pack(pady=10)

    tk.Button(bottom_frame, text="Add Blank Row", command=add_blank_row_user).pack(side=tk.LEFT, padx=4)
    tk.Button(bottom_frame, text="Save Table", command=save_user_table_to_csv).pack(side=tk.LEFT, padx=4)
    tk.Button(bottom_frame, text="Clear Table", command=clear_user_table).pack(side=tk.LEFT, padx=4)
    tk.Button(bottom_frame, text="Delete Selected Row", command=delete_selected_user_row).pack(side=tk.LEFT, padx=4)
    tk.Button(bottom_frame, text="Done", command=save_user_table).pack(side=tk.LEFT, padx=4)
    
    tk.Button(bottom_frame, text="Show Formula", command=lambda: show_formula_popup(
    r'$G = \mathrm{conductivity} \cdot \left( \dfrac{\mathrm{width} \cdot \mathrm{height}}{\mathrm{length}} \right) \cdot \mathrm{channel\_number}$',
    title="Theoretical Conductance Formula"
)).pack(side=tk.LEFT, padx=4)



    def compute_from_user_tree():
        try:
            length = nanochannel_dimensions.get("length")
            width = nanochannel_dimensions.get("width")
            height = nanochannel_dimensions.get("height")
            number = nanochannel_dimensions.get("number")
            if None in [length, width, height, number]:
                raise Exception("Nanochannel dimensions not set. Please click 'Done' first.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        results = []

        # 将数据按浓度从大到小排序
        rows = user_tree.get_children()
        data_sorted = sorted(
            [user_tree.item(row)['values'] for row in rows],
            key=lambda x: float(str(x[0]).replace("E", "e")),
            reverse=True
        )

        for values in data_sorted:
            try:
                conc = str(values[0])
                sigma = float(values[1])
                g = sigma * (width * height / length) * number
                results.append((conc, g))
            except Exception:
                continue

        # 弹出结果表格，不再在右侧输出
        open_theoretical_conductance_popup(results)

    tk.Button(bottom_frame, text="Compute Theoretical Conductance", command=compute_from_user_tree).pack(side=tk.LEFT, padx=4)


# 样例盐类数据（KCl 和 LiCl）
salt_data_dict = {
    "KCl": [("1", 9.83), ("0.1", 1.289), ("0.01", 0.14348), ("1E-3", 0.014688), ("1E-4", 0.0014979), ("1E-5", 0.00014979), ("1E-6", 0.000014979)],
    "LiCl": [("1", 6.34), ("0.1", 0.9581), ("0.01", 0.10727), ("1E-3", 0.011234), ("1E-4", 0.00115), ("1E-5", 0.000115), ("1E-6", 0.0000115)]
}






# %% [8] Main Function 3 (Core) : I-V Analysis
# %%% [8.1] Helper Function: Encloser Area Calculation Functions

# Open a new window with an interactive table store area data, (advanced functions:Input frequency, Plot Normoralised area against frequency in log scale) 
def open_area_table_window():
    '''
    Opens a new Toplevel window to display the "Enclosure Area Table".
    This table lists:
    - Filename: the name of the data file (e.g. IV curve)
    - Area (V·A = W): the enclosed area of the curve, representing energy or power
    - Frequency (Hz): the frequency used in the experiment, which can be edited

    Features:
    - Treeview table with column headers
    - Double-click any cell to edit its content using an Entry widget
    - Automatically saves the new value on Enter or discards on focus out
    '''
    # Create a new Toplevel window
    area_win = tk.Toplevel()
    area_win.title("Enclosure Area Table")

    # Define column names
    columns = ["Filename", "Area (V·A = W)", "Frequency (Hz)"]

    # Create a Treeview widget with the specified columns
    tree = ttk.Treeview(area_win, columns=columns, show="headings", selectmode="browse")

    # Set up each column's heading and width
    for col in columns:
        tree.heading(col, text=col)       # Set column heading text
        tree.column(col, width=200)       # Set column width

    # Pack the Treeview to fill the window and allow resizing
    tree.pack(fill=tk.BOTH, expand=True)

    # Define the double-click behavior for editing a cell
    def on_double_click(event):
        # Identify which row and column were clicked
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if item and column:
            col_index = int(column[1:]) - 1  # Convert "#1" to index 0
            if col_index < 0:
                return
            # Get the pixel location and size of the clicked cell
            x, y, width, height = tree.bbox(item, column)
            # Create an Entry widget positioned over the cell
            entry = tk.Entry(area_win)
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, tree.set(item, column))  # Pre-fill with current cell value
            # Define the function to save the new value on Enter
            def save_edit(event):
                tree.set(item, column, entry.get())  # Update Treeview with new value
                entry.destroy()                      # Remove the Entry widget
            # Bind Return key to save changes; unfocus to cancel
            entry.bind("<Return>", save_edit)
            entry.bind("<FocusOut>", lambda e: entry.destroy())

    # Bind double-click event to trigger the cell editing behavior
    tree.bind("<Double-1>", on_double_click)
    
    # Using data from the interactive area table (requires at least two valid rows), plot normalized area vs. frequency in log scale
    def plot_area_vs_log_freq():
        '''
        This function generates a plot of normalized area against frequency (log scale).
        It uses data from the Treeview widget (`tree`) filled by `open_area_table_window`.
   
        Main Features:
        - Filters rows with valid numerical area and positive frequency
        - Normalizes the area data between 0 and 1
        - Applies linear fitting: normalized_area = a * log10(freq) + b
        - Creates an interactive matplotlib plot inside a Tkinter window
        '''
        areas = [] # Store numeric area values
        freqs = [] # Store corresponding frequency values
        labels = [] # Store filenames (optional labels)
        # Extract valid area and frequency data from each row of the Treeview
        for row in tree.get_children(): 
            # Retrieve the values from this row as a list: [filename, area, frequency]
            values = tree.item(row)['values']
            try:
                area = float(values[1]) # Attempt to convert the second column (Area) to a float
                freq = float(values[2]) # Attempt to convert the third column (Frequency) to a float
                # Only include rows where frequency is positive (log scale requires freq > 0)
                if freq > 0:
                    areas.append(area) # Store the valid area value
                    freqs.append(freq) # Store the corresponding frequency
                    labels.append(values[0]) # Store the filename label (used for tooltips or traceability)
            except:
                continue  # Skip invalid or non-numeric entries

        # Ensure at least two valid data points are available for fitting
        if len(areas) < 2:
            messagebox.showerror("Error", "Please enter at least two valid area-frequency pairs.")
            return

        # Normalize area values
        area_min, area_max = min(areas), max(areas)
        normalized_areas = [(a - area_min) / (area_max - area_min) for a in areas]

        # Plot normalized area vs frequency (log scale)
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter_plot = ax.scatter(freqs, normalized_areas, color='blue', label="Data Points")
        fit_line, = ax.plot([], [], 'r--', label="Linear Fit")
        ax.set_xscale('log')
        ax.set_xlabel("Frequency [Hz] (log scale)")
        ax.set_ylabel("Normalized Area")
        ax.set_title("Normalized Area vs Frequency")
        ax.grid(True)
        ax.legend()

        # Perform linear regression on log10(frequency)
        coeffs = np.polyfit(np.log10(freqs), normalized_areas, 1)
        fit_y = np.polyval(coeffs, np.log10(freqs))
        fit_line.set_data(freqs, fit_y)

        # Open new Tkinter window to display the plot
        plot_win = tk.Toplevel(area_win)
        plot_win.title("Normalized Area vs Frequency")

        # Embed the plot into the window using FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Define button logic for controlling plot visibility
        def show_scatter_only():# Show only the scatter plot (hide the fit line)
            scatter_plot.set_visible(True) # Make scatter plot visible
            fit_line.set_visible(False) # Hide fit line
            canvas.draw() # Redraw the canvas to apply changes

        # Show only the fitted line (hide the scatter plot)
        def show_fit_only():
            scatter_plot.set_visible(False)
            fit_line.set_visible(True)
            canvas.draw()

        # Show both scatter plot and fitted line
        def show_both():
            scatter_plot.set_visible(True)
            fit_line.set_visible(True)
            canvas.draw()
        # Save the current data (freq, normalized area, fitted value) to CSV
        def save_data():
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
            if path:
                try:
                    import pandas as pd
                    df_out = pd.DataFrame({
                        "Filename": labels,
                        "Frequency_Hz": freqs,
                        "Normalized_Area": normalized_areas,
                        "Fit_Value": fit_y
                    })
                    df_out.to_csv(path, index=False)
                    messagebox.showinfo("Saved", f"Data saved to:\n{path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save data:\n{e}")

        # Save the current plot to a PNG file
        def save_plot():
            # Open a file dialog to let user choose save location and filename
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if path:
                try:
                    # Save the figure with high resolution and tight layout
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
                except Exception as e:
                    # Show an error message if saving fails
                    messagebox.showerror("Error", f"Failed to save plot:\n{e}")

        # Create a horizontal frame to hold control buttons
        button_frame = tk.Frame(plot_win)
        button_frame.pack(pady=5)

        # Add buttons to control plot visibility and save function
        tk.Button(button_frame, text="Scatter Only", command=show_scatter_only).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Fit Line Only", command=show_fit_only).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Fit + Scatter", command=show_both).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save Plot", command=save_plot).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Save Data", command=save_data).pack(side=tk.LEFT, padx=5)
    # Function to save the Treeview area table to a CSV file
    def save_table_data():
        # Open a file dialog to choose the output CSV file path
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not path:
            return # If user cancels, do nothing
        try:
            # Extract all rows from the Treeview as a list of value lists
            data = [tree.item(row)["values"] for row in tree.get_children()]
            # Create a pandas DataFrame from the data
            df_out = pd.DataFrame(data, columns=columns)
            # Save the DataFrame to the selected CSV file
            df_out.to_csv(path, index=False)
            # Show confirmation message after successful save
            messagebox.showinfo("Saved", f"Area table saved to:\n{path}")
        # Show confirmation message after successful save
        except Exception as e:
            # If saving fails, show error message with reason
            messagebox.showerror("Error", f"Failed to save table:\n{e}")

    # Create a horizontal frame to contain action buttons below the table
    btn_frame = tk.Frame(area_win)
    btn_frame.pack(pady=8)
    # Add a button to generate the Normalized Area vs Frequency plot
    tk.Button(btn_frame, text="Plot Normalized Area vs Frequency", command=plot_area_vs_log_freq).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Save Table as CSV", command=save_table_data).pack(side=tk.LEFT, padx=10)
    # Save references to the Treeview and window in the function object
    open_area_table_window.tree = tree
    open_area_table_window.window = area_win

def calculate_enclosure_area(selected_label, filepaths, v_unit_orig, v_unit, i_unit_orig, i_unit, unit_factors):
    """
    Hysteresis loop area using the same method as IV_loop.py:
    - De-duplicate adjacent points
    - Close the path
    - Compute union polygon area with pyclipper
    Units: V·A = W
    """
    import os, re
    import numpy as np
    import pandas as pd
    import tkinter as tk
    from tkinter import messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # ---- exact same core as your verified script ----
    def _compute_union_area_pyclipper(x, y):
        import pyclipper
        if len(x) < 3:
            return 0.0
        pts = list(zip(np.asarray(x, float), np.asarray(y, float)))

        # remove adjacent duplicates
        clean = [pts[0]]
        for p in pts[1:]:
            if p != clean[-1]:
                clean.append(p)
        # close path
        if clean[0] != clean[-1]:
            clean.append(clean[0])

        scale = 1e10
        ipts = [(int(round(px * scale)), int(round(py * scale))) for (px, py) in clean]
        pc = pyclipper.Pyclipper()
        pc.AddPath(ipts, pyclipper.PT_SUBJECT, True)
        sol = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        area = sum(pyclipper.Area(poly) for poly in sol) / (scale * scale)
        return abs(area)

    def _extract_filename(label: str):
        m = re.findall(r'([^\s()]+\.(?:xlsx|csv|txt))', str(label))
        return m[-1] if m else os.path.basename(str(label)).strip()

    try:
        # ---------- resolve path ----------
        wanted = _extract_filename(selected_label)
        matched_path = None
        for p in filepaths:
            base = os.path.basename(p)
            if base == wanted or wanted in base or base in str(selected_label):
                matched_path = p
                break
        if not matched_path:
            raise FileNotFoundError(f"Could not resolve file from label:\n“{selected_label}”")

        # ---------- load V, I ----------
        if matched_path.lower().endswith(".xlsx"):
            df = pd.read_excel(matched_path)
        else:
            df = pd.read_csv(matched_path, delim_whitespace=True, skiprows=2, header=None)
        if df.shape[1] < 2:
            raise ValueError("Input must have at least two columns (V, I).")

        V_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        I_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

        # clean NaNs
        m = (~np.isnan(V_raw)) & (~np.isnan(I_raw))
        V_raw, I_raw = V_raw[m], I_raw[m]
        if V_raw.size < 3:
            raise ValueError("Not enough valid points after cleaning.")

        # ---------- units ----------
        for k in (v_unit_orig, v_unit, i_unit_orig, i_unit):
            if k not in unit_factors:
                raise KeyError(f"Missing unit factor: {k}")
        V = V_raw * (unit_factors[v_unit_orig] / unit_factors[v_unit])
        I = I_raw * (unit_factors[i_unit_orig] / unit_factors[i_unit])

        # ---------- area (exactly your verified method) ----------
        area = _compute_union_area_pyclipper(V, I)

        # ---------- append to table (if present) ----------
        try:
            if (not hasattr(open_area_table_window, "tree")) or (not open_area_table_window.window.winfo_exists()):
                open_area_table_window()
            open_area_table_window.tree.insert("", "end",
                                               values=[os.path.basename(matched_path), f"{area:.7g}"])
        except Exception:
            pass

        # ---------- plot ----------
        pop = tk.Toplevel()
        pop.title(f"Hysteresis Area – {os.path.basename(matched_path)}")
        fig = Figure(figsize=(8.6, 3.8), dpi=100)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(V, I, linewidth=1.6)
        ax1.set_xlabel(f"Voltage ({v_unit})")
        ax1.set_ylabel(f"Current ({i_unit})")
        ax1.set_title("I–V Polyline")
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(V, I, linewidth=1.2, alpha=0.9, label="I–V")
        ax2.set_xlabel(f"Voltage ({v_unit})")
        ax2.set_ylabel(f"Current ({i_unit})")
        ax2.set_title(f"Union area (pyclipper) = {area:.7g} W")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc="best")

        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=pop).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        messagebox.showinfo("Hysteresis Area", f"Area in V–I plane = {area:.7g} V·A (W)")
        return area

    except Exception as e:
        messagebox.showerror("Area Calculation Error", str(e))
        return None






# %%% [8.2] Helper Function: Zoomed in Plot of Individual IV with Selected Region
# Functions to plot a zoomed in plot of individual IV with a selected region
def generate_zoomed_in_plot(ax_main, vmin, vmax, V, I, data_elements, v_unit, i_unit, G_full):
        # Get reference figure and DPI from main axis
        fig_main = ax_main.get_figure()
        dpi = fig_main.dpi
        # Get position and size of the main axis in figure coordinates
        bbox = ax_main.get_position()
        fig_width, fig_height = fig_main.get_size_inches()
        # Convert axis width and height to inches
        ax_width_in = bbox.width * fig_width
        ax_height_in = bbox.height * fig_height
        # Create a new figure and axis with the same size and DPI as main axis
        fig_zoom, ax_zoom = plt.subplots(figsize=(ax_width_in, ax_height_in), dpi=dpi)
        # Keep the same aspect ratio as the main axis
        ax_zoom.set_aspect(ax_main.get_aspect())
        # Match subplot padding to align with original axis
        fig_zoom.subplots_adjust(left=bbox.x0, bottom=bbox.y0, right=bbox.x1, top=bbox.y1)
        # Set zoom range for X and Y axes
        ax_zoom.set_xlim(vmin, vmax)
        ax_zoom.set_ylim(np.min(I[(V >= vmin) & (V <= vmax)]), np.max(I[(V >= vmin) & (V <= vmax)]))
        # Plot each segment within selected region
        for element in data_elements:
                if not hasattr(element, "get_xdata"):
                        continue
                xdata = element.get_xdata()
                ydata = element.get_ydata()
                color = element.get_color()
                label = element.get_label()
                # Select the data points within the zoom region
                mask = (xdata >= vmin) & (xdata <= vmax)
                x_zoom = xdata[mask]
                y_zoom = ydata[mask]
                # Plot if there are enough points
                if len(x_zoom) > 1:
                    if label and not label.startswith("_"): # Show label only if valid
                        ax_zoom.plot(x_zoom, y_zoom, color=color, label=label, linewidth=1.5)
                    else:
                        ax_zoom.plot(x_zoom, y_zoom, color=color, linewidth=1.5)
                # Set consistent plot style for zoomed region
                ax_zoom.set_title("Selected Region Fit", fontsize=13)
                ax_zoom.set_xlabel(f"Voltage ({v_unit})", fontsize=13)
                ax_zoom.set_ylabel(f"Current ({i_unit})", fontsize=13)
                ax_zoom.tick_params(labelsize=10)
                ax_zoom.grid(True, linestyle=':', alpha=0.6)
                ax_zoom.axhline(0, color='gray', linestyle='--')
                ax_zoom.axvline(0, color='gray', linestyle='--')
                
        # Linear regression for conductance in the selected region
        try:
                from scipy.stats import linregress
                mask_full = (V >= vmin) & (V <= vmax)
                V_selected = V[mask_full]
                I_selected = I[mask_full]
                slope, intercept, r_value, p_value, std_err = linregress(V_selected, I_selected)
                # Annotate conductance value on the top-left of zoomed plot
                text_x_zoom = ax_zoom.get_xlim()[0] + 0.05 * (ax_zoom.get_xlim()[1] - ax_zoom.get_xlim()[0])
                text_y_zoom = ax_zoom.get_ylim()[1] - 0.05 * (ax_zoom.get_ylim()[1] - ax_zoom.get_ylim()[0])
                ax_zoom.text(text_x_zoom, text_y_zoom,
                                f"Zoomed G = {slope:.2e} S",
                                fontsize=10, ha='left', va='top', color='black')
        except Exception as e:
                print("Linear regression failed:", e)

        # Interactive hover cursor (show current coordinates)
        import mplcursors
        cursor = mplcursors.cursor(ax_zoom, hover=True)
        @cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set_text(f"V = {x:.4g} {v_unit}\nI = {y:.4g} {i_unit}")
        # Return figure, axis, and computed slope
        return fig_zoom, ax_zoom, slope

# %%% [8.3] Helper Function: Finding I-V intersection
# Finding I-V intersection function
def find_self_intersection(V, I):
    """Find the point of self-intersection (or minimal loop distance) in a V-I curve.
    Parameters:
        V (ndarray): Voltage array
        I (ndarray): Current ardoray
    Returns:
        tuple: Coordinates (V, I) of the closest approach in the loop
    """
    points = np.column_stack((V, I))  # Combine V and I into 2D points
    n = len(points)
    min_dist = float('inf')  # Initialize minimum distance as infinity
    cross_idx = 0  # Index of crossing point
    for i in range(n):
        for j in range(i + 10, n):  # Skip close neighbors to avoid trivial minima
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist
                cross_idx = i
    return V[cross_idx], I[cross_idx]  # Return the voltage and current at closest point

# %%% [8.4] Helper Function: Load and Read Data
# Read data function
def load_data(filepath):
    """Load voltage-current data from an Excel or text file.
    Parameters:
        filepath (str): Path to the data file
    Returns:
        tuple: Voltage array (V), Current array (I)
    """
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)  # Load Excel file
    else:
        df = pd.read_csv(filepath, sep=None, engine="python", skiprows=2, header=None)  # Load .txt file, skip header
        df = df.dropna()  # Remove rows with NaN values
    if df.shape[1] < 2:
        raise ValueError("File must contain at least two numeric columns")
    try:
        V = df.iloc[:, 0].astype(float).to_numpy()  # First column: Voltage
        I = df.iloc[:, 1].astype(float).to_numpy()  # Second column: Current
    except Exception as e:
        raise ValueError(f"Data parsing error: {e}")
    return V, I  # Return voltage and current arrays

# %%% [8.5] Helper Function: Experimental C, G, σ Analysis and combination
# IV addisional helper function, calculation of experimental conductivity, and combine concentration, experimental conductance and experimental conductivity
def save_combined_conductance_conductivity():
    """
    Save a CSV file containing both experimental conductance and derived conductivity data
    for all uploaded IV curves, based on known nanochannel geometry.
    The conductivity is calculated using:
        σ = G × L / (A × N)
    where:
        - G: experimental conductance (S)
        - L: nanochannel length (m)
        - A: nanochannel cross-sectional area = width × height (m²)
        - N: number of channels

    The saved table includes:
        - Filename
        - Concentration
        - Conductance (S)
        - Conductivity (S/m)

    Sorting is applied based on descending concentration for easier comparison.
    """
    try:
        # If no conductance data is available, exit early
        if not conductance_table:
            messagebox.showinfo("Info", "No data to save.")
            return
        # Attempt to retrieve and validate nanochannel geometric dimensions
        try:
            length = float(nanochannel_dimensions["length"])
            width = float(nanochannel_dimensions["width"])
            height = float(nanochannel_dimensions["height"])
            number = int(nanochannel_dimensions["number"])
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid nanochannel dimensions:\n{e}")
            return
        
        # Compute nanochannel cross-sectional area
        area = width * height
       
        # Prepare table rows: [filename, concentration, conductance, conductivity]
        rows = []
        for filepath, (conc, G) in conductance_table.items():
            try:
                conc_sort = float(conc) # Used for sorting
            except:
                conc_sort = None
            # Compute conductivity from conductance
            sigma = G * length / (area * number)
            rows.append([os.path.basename(filepath), conc, G, sigma, conc_sort])
        
        # Convert to DataFrame and sort by concentration (descending)
        df = pd.DataFrame(rows, columns=["Filename", "Concentration", "Conductance (S)", "Conductivity (S/m)", "_SortKey"])
        df.sort_values(by="_SortKey", ascending=False, inplace=True)
        df.drop(columns=["_SortKey"], inplace=True)
        
        # Ask user where to save the CSV file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV File", "*.csv")],
                                                 title="Save Experimental Results")
        # Save if user confirmed path
        if file_path:
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Saved", f"Results saved to:\n{file_path}")

    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to save results:\n{e}")

# %%% [8.6] Helper Function: Function to add an additional IV curve to an existing plot
# IV additional helper fucntion 2: Function to add an additional IV curve to an existing plot without clearing it
def add_additional_iv_curve():
    '''
    Add an additional IV (Current-Voltage) curve to the existing plot without clearing previous curves.

    This function:
    - Prompts the user to select a file containing IV data (.xlsx or .txt)
    - Loads voltage and current values, applies unit conversions
    - Calculates conductance (slope) via linear regression
    - Automatically assigns unique colors to up-sweep and down-sweep segments
    - Detects crossover point (I ≈ 0) and labels the conductance value
    - Updates the plot by adding the new curve and refreshing the legend
    - Stores metadata for the curve for later manipulation (e.g., deletion, legend updates)

    Data is also saved to `conductance_table` for later export or conductivity calculation.
    '''
    try:
        # Retrieve previously stored plot objects and state
        ax = load_files_and_compute_conductance.current_ax # Active plot axis
        fig = load_files_and_compute_conductance.current_fig # Active matplotlib figure
        curves = load_files_and_compute_conductance.curves # Dictionary storing existing curves
        color_set = load_files_and_compute_conductance.color_set # Used to track unique colors
        v_to = load_files_and_compute_conductance.v_unit # Target voltage unit
        i_to = load_files_and_compute_conductance.i_unit # Target current unit
        # Assign a unique ID to this new curve
        curve_id = load_files_and_compute_conductance.curve_id_counter
        load_files_and_compute_conductance.curve_id_counter += 1
        # Ask user to select a new file to add 
        filepath = filedialog.askopenfilename(filetypes=[("Excel and Text files", "*.xlsx *.txt")])
        if not filepath:
            return # If user cancels, do nothing
        # Load voltage-current data from selected file
        if filepath.endswith(".xlsx"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)
        # Extract raw voltage and current columns as NumPy arrays
        V_raw = df.iloc[:, 0].to_numpy(dtype=float)
        I_raw = df.iloc[:, 1].to_numpy(dtype=float)
        # Perform unit conversion to target voltage/current units
        v_from = load_files_and_compute_conductance.v_unit_orig.get()
        i_from = load_files_and_compute_conductance.i_unit_orig.get()
        V = V_raw * (unit_factors[v_from] / unit_factors[v_to])
        I = I_raw * (unit_factors[i_from] / unit_factors[i_to])
        # Compute conductance using linear regression (slope of I-V)
        slope, _, _, _, _ = linregress(V, I)
        # Extract conductance and update conductance_table
        try:
            match = re.search(r"(\d+\.?\d*(?:[eE][-+]?\d+)?)", os.path.basename(filepath))

            concentration = float(match.group(1)) if match else None
            if concentration is not None and filepath not in conductance_table:
                conductance_table[filepath] = (concentration, slope)
        except:
            pass

        # Find the point closest to I = 0 (used as crossover)
        #cross_idx = np.argmin(np.abs(I))
        #cross_V = V[cross_idx]
        #cross_I = I[cross_idx]
        # Assign unique colors for up- and down-sweep segments
        def random_unique_color(existing_colors):
            while True:
                # Generate a random RGB color that is not already in the set of used colors
                c = (random.random(), random.random(), random.random()) # Randomly generate an RGB tuple: (R, G, B), each in [0, 1]
                if c not in existing_colors:
                    return c # If this color has not been used yet, return it
        # Generate a unique color for up-sweep segments of the curve
        color_up = random_unique_color(color_set)
        color_set.add(color_up) # Add it to the set of used colors to prevent duplicates
        # Generate another unique color for down-sweep segments of the curve
        color_down = random_unique_color(color_set)
        color_set.add(color_down) # Ensure future curves won't reuse the same color
        # Plot curve with color-coded segments
        curve_elements = []
        for i in range(len(V) - 1):
            is_down_sweep = V[i + 1] < V[i]
            color = color_down if is_down_sweep else color_up
            style = '--' if is_down_sweep else '-'
            line = ax.plot(V[i:i + 2], I[i:i + 2], color=color, linestyle=style, linewidth=1.5)[0]
            curve_elements.append(line)

        '''
        # Plot crossover point (near I=0) as yellow dot
        pt = ax.plot(cross_V, cross_I, 'o', markersize=8,
                     markerfacecolor='yellow', markeredgecolor='black',
                     label=f'Crossover {os.path.basename(filepath)}')[0]
        curve_elements.append(pt)
        '''
        # Add dummy lines for legend (for up/down-sweep labels)
        leg1 = ax.plot([], [], color=color_up, linestyle='-', label=f"Up-sweep ({os.path.basename(filepath)})")[0]
        leg2 = ax.plot([], [], color=color_down, linestyle='--', label=f"Down-sweep ({os.path.basename(filepath)})")[0]

        curve_elements.extend([leg1, leg2])
        # Add conductance label in upper left of plot
        txt = ax.text(0.02, 0.98 - 0.07 * len(curves),
                     f"{os.path.basename(filepath)}\nG = {slope:.2e} S", # Dynamically offset based on number of existing labels
                     transform=ax.transAxes, fontsize=9, color='black',
                     verticalalignment='top')
        curve_elements.append(txt)
         # Save this new curve and its metadata into the curves dictionary
        curves[curve_id] = {
            "elements": curve_elements,
            "label": os.path.basename(filepath),
            "filepath": filepath  # Critical for area calculation later
        }
        # Redraw the canvas to show the newly added curve
        fig.canvas.draw()
        # Refresh legend to include new handles
        if hasattr(ax, "legend_") and ax.legend_:
            ax.legend_.remove()
            ax.legend(handles=[h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")],
                     loc="upper right")
            fig.canvas.draw()
        # Update the curve selection dropdown menu if available
        if hasattr(load_files_and_compute_conductance, "curve_selector"):
            load_files_and_compute_conductance.curve_selector['values'] = [
                data["label"] for data in load_files_and_compute_conductance.curves.values()
            ]
            load_files_and_compute_conductance.curve_selector.set('')
    except Exception as e:
        # Catch any error and show a user-friendly message
        messagebox.showerror("Error", f"Failed to add IV curve\n{e}")

# %%% [8.7] Core Function: **MAIN IV-Analysis Function** of I-V Analysis
# Main function for I-V analysis, I-V plot and conductance calculating function (Advanced ability: determine sweep direction; add or delete IV curves to existing plot)
def load_files_and_compute_conductance(rescale=False):
    """
    Load multiple IV data files, apply unit conversion, compute conductance via linear fit,
    and plot the IV curves with interactive and exportable options.

    Advanced Features:
    - Handles multiple file types (.xlsx, .txt)
    - Supports unit conversion for voltage/current axes
    - Computes conductance (slope) using linear regression
    - Detects crossover (self-intersection) point in the curve
    - Automatically separates up-sweep and down-sweep using color
    - Allows adding and deleting curves interactively via GUI

    Parameters:
        rescale (bool): 
            - If False: open file dialog to load new IV data files.
            - If True: re-apply plotting with updated unit settings (e.g., after unit rescaling).
    """
    try:
        if not rescale:
            # Open file dialog to select IV data files (Excel or whitespace-delimited text)
            load_files_and_compute_conductance.filepaths = filedialog.askopenfilenames(
                filetypes=[("Excel and Text files", "*.xlsx *.txt")])
            # If user cancels file selection, exit
            if not load_files_and_compute_conductance.filepaths:
                return
            # Initialize the unit conversion UI when files are first loaded
            # Always recreate the unit row (gives us a fresh zoom_block_frame)
            attach_unit_conversion_ui(right_frame, load_files_and_compute_conductance)  # NEW/CHANGED

        # Retrieve user-selected original and target units for voltage and current
        v_from = load_files_and_compute_conductance.v_unit_orig.get()
        v_to = load_files_and_compute_conductance.v_unit_target.get()
        i_from = load_files_and_compute_conductance.i_unit_orig.get()
        i_to = load_files_and_compute_conductance.i_unit_target.get()
    
        # Clear previous plot area (or existing UI region if any)）
        if hasattr(load_files_and_compute_conductance, "plot_frame") and load_files_and_compute_conductance.plot_frame.winfo_exists():
            load_files_and_compute_conductance.plot_frame.destroy()

        # Remove all widgets from the right_frame to avoid overlapping UIs
        for widget in right_frame.winfo_children():
            widget.pack_forget()

        # Always refresh the unit conversion UI (both on first load and during rescale)
        attach_unit_conversion_ui(right_frame, load_files_and_compute_conductance)
        
        # Create a new plot_frame that fills the right panel
        load_files_and_compute_conductance.plot_frame = tk.Frame(right_frame)
        load_files_and_compute_conductance.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(8, 4))
        color_set = set()
        curves = {}
        curve_id_counter = 0
        # Loop through each selected file path
        for filepath in load_files_and_compute_conductance.filepaths:
            # Load data from Excel or text file
            if filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)
            # Extract voltage and current columns
            V_raw = df.iloc[:, 0].to_numpy(dtype=float)
            I_raw = df.iloc[:, 1].to_numpy(dtype=float)
            # Apply unit conversion to match target display units
            V = V_raw * (unit_factors[v_from] / unit_factors[v_to])
            I = I_raw * (unit_factors[i_from] / unit_factors[i_to])
            # Perform linear regression: I = slope * V + intercept
            slope, _, _, _, _ = linregress(V, I)
            # Extract concentration from filename (e.g., 100mM_KCl.xlsx → 100)
            match = re.search(r"(\d+\.?\d*(?:[eE][-+]?\d+)?)", os.path.basename(filepath))
            concentration = match.group(1) if match else "N/A"
            
            # Only add if not already stored
            if filepath not in conductance_table:
                conductance_table[filepath] = (concentration, slope)
            # Find the self-intersection point (if any), often used as crossover point
            cross_V, cross_I = find_self_intersection(V, I)

            # Helper function: generate a random RGB color not already in the used color set
            def random_unique_color(existing_colors):
                while True:
                    c = (random.random(), random.random(), random.random()) # RGB tuple
                    if c not in existing_colors:
                        return c
            # Assign a unique color for upward sweep
            color_up = random_unique_color(color_set)
            color_set.add(color_up)
            # Assign another unique color for downward sweep
            color_down = random_unique_color(color_set)
            color_set.add(color_down)
            curve_elements = []# List to store all visual elements related to this curve
            # Draw the IV curve segment-by-segment; use up/down color based on slope direction
            # Draw the IV curve segment-by-segment; use up/down color and line style
            for i in range(len(V) - 1):
                is_down_sweep = V[i + 1] < V[i]
                color = color_up if not is_down_sweep else color_down
                style = '--' if is_down_sweep else '-'
                line = ax.plot(V[i:i + 2], I[i:i + 2], color=color, linestyle=style, linewidth=1.5)[0]
                curve_elements.append(line)

            # Plot the crossover (self-intersection) point as a yellow dot
            '''
            pt = ax.plot(cross_V, cross_I, 'o', markersize=8,
                         markerfacecolor='yellow', markeredgecolor='black',
                         label=f'Crossover {os.path.basename(filepath)}')[0]
            curve_elements.append(pt)
            '''
            # Create legend handles (empty lines) for up/down sweep
            leg1 = ax.plot([], [], color=color_up, linestyle='-', label=f"Up-sweep ({os.path.basename(filepath)})")[0]
            leg2 = ax.plot([], [], color=color_down, linestyle='--', label=f"Down-sweep ({os.path.basename(filepath)})")[0]

            curve_elements.extend([leg1, leg2])
            # Add text label in top-left showing filename and calculated conductance
            txt = ax.text(0.02, 0.98 - 0.07 * len(ax.texts),#(location of the text)
                          f"{os.path.basename(filepath)}\nG = {slope:.2e} S",
                          transform=ax.transAxes, fontsize=9, color='black',
                          verticalalignment='top')
            curve_elements.append(txt)
            # Store all components in the curve registry (dictionary)
            curves[curve_id_counter] = {
                "elements": curve_elements,
                "label": os.path.basename(filepath),
                "filepath": filepath
                }
            curve_id_counter += 1 # Increment ID for next curve

        # Draw horizontal and vertical reference lines at V=0 and I=0
        ax.axhline(0, color='black', linestyle=':', linewidth=0.7)
        ax.axvline(0, color='black', linestyle=':', linewidth=0.7)
        # Set axis labels and plot title with correct units
        ax.set_xlabel(f"Voltage ({v_to})")
        ax.set_ylabel(f"Current ({i_to})")
        ax.set_title("IV Curve(s)")

        # Remove old legend if it exists (to prevent duplication)
        if hasattr(ax, "legend_") and ax.legend_:
            ax.legend_.remove()
        ax.legend(handles=[h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")], loc="upper right")
        ax.grid(True, linestyle=':', alpha=0.5) # Light dotted grid for readability
        fig.tight_layout() # Auto-fit elements to figure size
        # Embed the matplotlib figure into the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=load_files_and_compute_conductance.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add interactive hover tooltip using mplcursors
        mplcursors.cursor(ax.lines, hover=True).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"V = {sel.target[0]:.4g} {v_to}\nI = {sel.target[1]:.4g} {i_to}"))
        # Save current figure, axis, and metadata to the function's state
        load_files_and_compute_conductance.current_ax = ax
        load_files_and_compute_conductance.current_fig = fig
        load_files_and_compute_conductance.color_set = color_set
        load_files_and_compute_conductance.curves = curves
        load_files_and_compute_conductance.curve_id_counter = curve_id_counter
        load_files_and_compute_conductance.v_unit = v_to
        load_files_and_compute_conductance.i_unit = i_to
        # === Attach zoom controls now that fig/ax exist ===
        attach_zoom_controls(
            load_files_and_compute_conductance.zoom_block_frame,  # NEW/CHANGED
            load_files_and_compute_conductance.current_fig,
            load_files_and_compute_conductance.current_ax
            )
        # Create a control panel below the plot with buttons and dropdown
        control_frame = tk.Frame(load_files_and_compute_conductance.plot_frame)
        control_frame.pack(pady=8)
        # Button to add another IV curve without clearing the current plot
        tk.Button(control_frame, text="Add IV to Existing Plot", command=add_additional_iv_curve).pack(side=tk.LEFT, padx=5)
        # Label and dropdown (Combobox) for selecting curve to delete
        tk.Label(control_frame, text="Select IV to Analyze:").pack(side=tk.LEFT, padx=(20, 5))
        curve_selector = ttk.Combobox(control_frame, state="readonly", width=30)
        curve_selector.pack(side=tk.LEFT, padx=5)
        # Store reference to the selector widget so it can be updated externally
        load_files_and_compute_conductance.curve_selector = curve_selector
        
        # Function to delete the IV curve selected in the dropdown menu
        def delete_selected_iv_curve():
            selected_label = curve_selector.get() # Get the currently selected curve label from the dropdown menu
            if not selected_label:
                # If nothing is selected, inform the user
                messagebox.showinfo("Info", "Please select an IV curve to delete.")
                return
            to_delete = None # Will store the ID of the curve to remove
            # Search for the curve ID whose label matches the selected label
            for cid, data in load_files_and_compute_conductance.curves.items():
                if data["label"] == selected_label:
                    to_delete = cid
                    break
            # If a matching curve is found
            if to_delete is not None:
                # Loop through all plot elements associated with this curve (lines, dots, text)
                for element in load_files_and_compute_conductance.curves[to_delete]["elements"]:
                    try:
                        # Try to remove each element from the plot
                        element.remove()
                    except:
                        # Some elements (like text) may not support .remove(), so safely ignore
                        pass
                    
                # Remove the curve data from the internal dictionary
                del load_files_and_compute_conductance.curves[to_delete]
                # If a legend exists, remove it to refresh it later
                if hasattr(ax, "legend_") and ax.legend_:
                    ax.legend_.remove()
                # Rebuild the legend using only visible, labeled lines
                ax.legend(handles=[h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")], loc="upper right")
                # Redraw the canvas to reflect removal
                fig.canvas.draw()
                # Update the dropdown menu to reflect remaining curves
                curve_selector['values'] = [data["label"] for data in load_files_and_compute_conductance.curves.values()]
                curve_selector.set('') # Clear the selection in the dropdown after deletion
                
        # Function to remove all IV curves from the plot and reset the curve registry
        def clear_all_iv_curves():
            # Iterate over all stored curve IDs (create a list to avoid runtime dict mutation)
            for cid in list(load_files_and_compute_conductance.curves):
                # Remove all visual elements (lines, crossover point, text, legend handles)
                for element in load_files_and_compute_conductance.curves[cid]["elements"]:
                    try:
                        element.remove() # Attempt to remove the element from the plot
                    except:
                        pass # Some elements (e.g., text) may not support .remove(), safely skip
                # Delete the curve entry from the internal curves dictionary
                del load_files_and_compute_conductance.curves[cid]

            # If a legend is currently displayed, remove it before rebuilding
            if hasattr(ax, "legend_") and ax.legend_:
                ax.legend_.remove()
            # Rebuild the legend from remaining visible labeled lines (if any)
            ax.legend(handles=[h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")], loc="upper right")
            # Redraw the figure to reflect the removal of all curves
            fig.canvas.draw()
            curve_selector['values'] = [] # Clear all items from the curve selection dropdown
            curve_selector.set('') # Reset the dropdown selection to blank

        
        # Function to view selected IV curve
        conductance_table_window = None
        conductance_tree = None
        def view_selected_iv_curve():
                '''
                This function is used to:
                - View a selected I–V curve in a new window.
                - Plot both Up- and Down-sweeps with different line styles.
                - Calculate full-region and sub-region conductance.
                - Display a zoomed-in version of the selected region.
                - Add results to an editable table.
                - Allow interactive saving, zooming, and cursor hover info.
                '''
                # Get the selected curve label from dropdown
                selected_label = curve_selector.get()
                if not selected_label:
                        messagebox.showinfo("Info", "Please select an IV curve to view.")
                        return
                # Match the selected label to a loaded curve's ID
                for cid, data in load_files_and_compute_conductance.curves.items():
                        if data["label"] == selected_label:
                                break
                else:
                        messagebox.showerror("Error", "Selected curve not found.")
                        return
                # Retrieve the filepath to load raw IV data
                curve_filepath = load_files_and_compute_conductance.curves[cid]["filepath"]
                # Create a new popup window for displaying the selected IV curve
                win = tk.Toplevel()
                win.title(f"IV Curve: {selected_label}")
                # Frame to contain all content
                view_frame = tk.Frame(win)
                view_frame.pack(fill=tk.BOTH, expand=True)
                # Subframe to hold matplotlib plots
                plot_container = tk.Frame(view_frame)
                plot_container.pack(fill=tk.BOTH, expand=True)
                # Create the main matplotlib figure for full IV curve
                fig_view, ax_view = plt.subplots(figsize=(6, 4))
                # Replot the up/down sweeps with different styles for clarit
                for element in data["elements"]:
                        if isinstance(element, plt.Line2D):
                                xdata = element.get_xdata()
                                ydata = element.get_ydata()
                                color = element.get_color()
                                label = element.get_label()
                                # Solid line for Up-sweep, dashed for Down-sweep
                                if label.startswith("Up-sweep"):
                                    ax_view.plot(xdata, ydata, color=color, linestyle='-', label=label)
                                elif label.startswith("Down-sweep"):
                                        ax_view.plot(xdata, ydata, color=color, linestyle='--', label=label)
                                else:
                                        ax_view.plot(xdata, ydata, color=color)
                        # Crossover point (intersection) displayed as yellow circle
                        elif isinstance(element, plt.Artist) and hasattr(element, "get_markerfacecolor"):
                                xdata = element.get_xdata()
                                ydata = element.get_ydata()
                                ax_view.plot(xdata, ydata, 'o', color='yellow', markeredgecolor='black', label='Crossover')

                # Add horizontal and vertical axis lines at 0
                ax_view.axhline(0, color='gray', linestyle='--')
                ax_view.axvline(0, color='gray', linestyle='--')
                # Set labels and title
                ax_view.set_xlabel(f"Voltage ({load_files_and_compute_conductance.v_unit})")
                ax_view.set_ylabel(f"Current ({load_files_and_compute_conductance.i_unit})")
                ax_view.set_title(f"Individual IV Curve\n{selected_label}")
                ax_view.legend()
                ax_view.grid(True, linestyle=':', alpha=0.6)
                fig_view.tight_layout()
                # Load the raw IV data from file (before any rescaling)
                df = pd.read_excel(curve_filepath)
                V_raw = df.iloc[:, 0].to_numpy(dtype=float)
                I_raw = df.iloc[:, 1].to_numpy(dtype=float)
                # Perform unit conversion using original & display units
                v_from = load_files_and_compute_conductance.v_unit_orig.get()
                v_to = load_files_and_compute_conductance.v_unit
                i_from = load_files_and_compute_conductance.i_unit_orig.get()
                i_to = load_files_and_compute_conductance.i_unit
                V = V_raw * (unit_factors[v_from] / unit_factors[v_to])
                I = I_raw * (unit_factors[i_from] / unit_factors[i_to])
                # Fit full IV range with linear regression to get total conductance
                from scipy.stats import linregress
                slope_full, _, _, _, _ = linregress(V, I)
                data["conductance"] = slope_full
                # Annotate the full conductance on bottom-right of main plot
                x0, x1 = ax_view.get_xlim()
                y0, y1 = ax_view.get_ylim()
                pad = 0.05  # 5% inset from edges
                
                text_x_main = x1 - pad * (x1 - x0)   # right side
                text_y_main = y0 + pad * (y1 - y0)   # bottom side
                
                ax_view.text(
                    text_x_main, text_y_main,
                    f"Full G = {slope_full:.2e} S",
                    ha='right', va='bottom',
                    fontsize=12, color='black'
                    )

                # Embed main figure in GUI
                canvas = FigureCanvasTkAgg(fig_view, master=plot_container)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Input boxes for sub-region voltage range (V_max above V_min)
                input_frame = tk.Frame(view_frame)
                input_frame.pack(pady=6)

                tk.Label(input_frame, text="V_max:").grid(row=0, column=0, sticky="e")
                entry_vmax = tk.Entry(input_frame, width=10)
                entry_vmax.grid(row=0, column=1, padx=5)

                tk.Label(input_frame, text="V_min:").grid(row=1, column=0, sticky="e")
                entry_vmin = tk.Entry(input_frame, width=10)
                entry_vmin.grid(row=1, column=1, padx=5)


                # Subregion fitting logic: calculate conductance in user-defined voltage range
                def calculate_selected_region():
                        try:
                                # Read user-specified voltage limits from entry boxes
                                vmin = float(entry_vmin.get()) # Minimum voltage for fitting range
                                vmax = float(entry_vmax.get()) # Maximum voltage for fitting range
                                # Create mask to filter voltage and current arrays within this range
                                mask = (V >= vmin) & (V <= vmax) # Boolean mask for selecting data within range
                                V_region = V[mask] # Voltage values in selected region
                                I_region = I[mask] # Current values in selected region
                                # Check if enough data points exist in the region for regression
                                if len(V_region) < 2:
                                        messagebox.showerror("Error", "Not enough points in selected region.")
                                        return
                                
                                # Perform linear regression on selected region
                                slope, _, _, _, _ = linregress(V_region, I_region) # Calculate conductance from slope
                                G_full = data["conductance"] # Full-range conductance for reference
                                # Generate the zoom-in figure using pre-defined function
                                fig_zoom, ax_zoom, _ = generate_zoomed_in_plot(
                                        ax_view, vmin, vmax, V, I,
                                        data["elements"], v_to, i_to, G_full)

                                # Prepare table to store results
                                nonlocal conductance_table_window, conductance_tree ## Use previously declared globals (just above View_Selected_IV_Curve equation)
                                # If table already exists, use it; otherwise create a new pop-up window and Treeview
                                if conductance_table_window and conductance_table_window.winfo_exists():
                                    tree = conductance_tree # Reuse existing table
                                else:
                                    # Create new top-level window to hold the table
                                    conductance_table_window = tk.Toplevel(view_frame)
                                    conductance_table_window.title("Experimental Conductance in Selected Region")
                                    # Create a new Treeview table with 3 columns
                                    tree = ttk.Treeview(conductance_table_window,
                                                        columns=("Filename", "Voltage Range", "Conductance (S)"),
                                                        show="headings")
                                    # Set table column headers
                                    tree.heading("Filename", text="Filename")
                                    tree.heading("Voltage Range", text="Voltage Range")
                                    tree.heading("Conductance (S)", text="Conductance (S)")
                                    # Add table to the window
                                    tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
                                    # Save reference globally for reuse
                                    conductance_tree = tree

                                    # Make table editable on double-click
                                    def on_double_click(event):
                                        # Get selected row ID
                                        item_id = tree.focus()
                                        # Determine which column was clicked
                                        column = tree.identify_column(event.x)
                                        # Convert "#1" to index 0
                                        col_index = int(column[1:]) - 1  # "#1" → 0
                                        if item_id:
                                            old_val = tree.item(item_id, 'values')[col_index] # Get current cell value
                                            # Create temporary Entry box in place for editing
                                            entry_popup = tk.Entry(conductance_table_window)
                                            entry_popup.insert(0, old_val)
                                            entry_popup.focus()
                                            # Position popup over clicked cell
                                            x, y, width, height = tree.bbox(item_id, column)
                                            entry_popup.place(x=x, y=y, width=width, height=height)
                                            # Save edited value on Enter key
                                            def save_edit(event):
                                                new_val = entry_popup.get()
                                                values = list(tree.item(item_id, 'values')) # All row values
                                                values[col_index] = new_val # Replace only the edited column
                                                tree.item(item_id, values=values) # Update row
                                                entry_popup.destroy()
                                            # Bind Enter key and focus-out to trigger save
                                            entry_popup.bind("<Return>", save_edit)
                                            entry_popup.bind("<FocusOut>", lambda e: entry_popup.destroy())
                                    # Bind double-click event to editing function
                                    tree.bind("<Double-1>", on_double_click)

                                    # Add save table to CSV option
                                    def save_conductance_table():
                                        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                                            filetypes=[("CSV Files", "*.csv")])
                                        if path:
                                            try:
                                                # Gather all table rows into a list of lists
                                                rows = [tree.item(child)["values"] for child in tree.get_children()]
                                                df_save = pd.DataFrame(rows, columns=["Filename", "Voltage Range", "Conductance (S)"])
                                                df_save.to_csv(path, index=False)  # Save as CSV
                                                messagebox.showinfo("Saved", f"Table saved to:\n{path}")
                                            except Exception as e:
                                                messagebox.showerror("Error", f"Failed to save table:\n{e}")
                                    # Add save button to window
                                    tk.Button(conductance_table_window,
                                              text="Save Table as CSV",
                                              command=save_conductance_table).pack(pady=5)

                                # Add new row to the table
                                tree.insert("", "end", values=(selected_label, f"{vmin:.2f} to {vmax:.2f}", f"{slope:.2e}"))
                                # Display right-side zoomed-in plot
                                canvas_zoom = FigureCanvasTkAgg(fig_zoom, master=plot_container)
                                canvas_zoom.draw()
                                canvas_zoom.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6)
                                # Plot saving buttons for left & right plot
                                save_frame = tk.Frame(view_frame)
                                save_frame.pack(pady=6)

                                # Save left (main IV) plot as PNG
                                def save_left_plot():
                                    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
                                    if path:
                                        try:
                                            fig_view.savefig(path, dpi=300, bbox_inches='tight')
                                        except Exception as err:
                                            messagebox.showerror("Save Error", f"Failed to save left plot:\n{err}")
                                            
                                # Save right (zoomed-in) plot as PNG
                                def save_right_plot():
                                    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
                                    if path:
                                        try:
                                            fig_zoom.savefig(path, dpi=300, bbox_inches='tight')
                                        except Exception as err:
                                            messagebox.showerror("Save Error", f"Failed to save right plot:\n{err}")

                                # Add both buttons to UI
                                tk.Button(save_frame, text="Save Left Plot", command=save_left_plot).pack(side=tk.LEFT, padx=10)
                                tk.Button(save_frame, text="Save Right Plot", command=save_right_plot).pack(side=tk.LEFT, padx=10)

                        # Catch-all for unexpected runtime errors
                        except Exception as e:
                                messagebox.showerror("Error", f"Failed to compute conductance:\n{e}")
                # Add trigger button to GUI to launch this function
                tk.Button(view_frame, text="Calculate Conductance in Selected Region", command=calculate_selected_region).pack(pady=6)

                # Save the main IV curve figure (left side) as a PNG image
                def save_individual_plot():
                        # Open file dialog for user to choose file name and location
                        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")]) # Default extension for image file
                        if path: # Proceed only if a valid path is chosen
                                try:
                                        # Save the full-size IV figure using high DPI and tight bounding box
                                        fig_view.savefig(path, dpi=300, bbox_inches='tight')
                                except Exception as err:
                                        # Show error popup if saving fails
                                        messagebox.showerror("Save Error", f"Failed to save plot:\n{err}")

                # Add zoom controls to the IV plot
                zoom_frame = tk.Frame(win) # Create a frame to hold zoom buttons
                zoom_frame.pack(pady=6) # Add padding and place it in the main window
                # Call the shared zoom control function, passing current figure and axis
                attach_zoom_controls(zoom_frame, fig_view, ax_view)
                # Add "Save Plot" button below the zoom controls
                tk.Button(win, text="Save Plot", command=save_individual_plot).pack(pady=6)
                tk.Button(win, text="Calculate Enclosure Area", command=lambda: calculate_enclosure_area(
                        os.path.basename(curve_filepath),
                        [curve_filepath],
                        load_files_and_compute_conductance.v_unit_orig.get(),
                        load_files_and_compute_conductance.v_unit,
                        load_files_and_compute_conductance.i_unit_orig.get(),
                        load_files_and_compute_conductance.i_unit,
                        unit_factors
                )).pack(pady=4)

                mplcursors.cursor(ax_view.lines, hover=True).connect(
                        "add", lambda sel: sel.annotation.set_text(
                                f"V = {sel.target[0]:.4g} {load_files_and_compute_conductance.v_unit}\n"
                                f"I = {sel.target[1]:.4g} {load_files_and_compute_conductance.i_unit}"
                        )
                )


        # To view and analysis G-V from one of the uploaded IV
        def view_selected_gv_curve():
                """
                Display an individual G-V (Conductance vs Voltage) curve for the selected IV dataset.
                The conductance is calculated as G = I / V (excluding V=0), and an upper/lower envelope is generated.
                A separate window opens to plot the G-V curve with interactive cursor and save option.
                """
                # Get the selected curve label from the dropdown
                selected_label = curve_selector.get()
                if not selected_label:
                        messagebox.showinfo("Info", "Please select a curve to view.")
                        return
                
                # Find the matching curve by label
                for cid, data in load_files_and_compute_conductance.curves.items():
                        if data["label"] == selected_label:
                                break
                else:
                        messagebox.showerror("Error", "Selected curve not found.")
                        return

                # Load raw data from the original file
                filepath = data["filepath"]
                if filepath.endswith(".xlsx"):
                        df = pd.read_excel(filepath)
                else:
                        df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)
                
                # Extract voltage and current columns
                V_raw = df.iloc[:, 0].to_numpy(dtype=float)
                I_raw = df.iloc[:, 1].to_numpy(dtype=float)
                # Apply unit conversion based on selected units
                v_from = load_files_and_compute_conductance.v_unit_orig.get()
                v_to = load_files_and_compute_conductance.v_unit
                i_from = load_files_and_compute_conductance.i_unit_orig.get()
                i_to = load_files_and_compute_conductance.i_unit
                V = V_raw * (unit_factors[v_from] / unit_factors[v_to])
                I = I_raw * (unit_factors[i_from] / unit_factors[i_to])
                # Remove points where V is zero (to avoid division by zero)
                mask = ~np.isclose(V, 0)
                V_valid, I_valid = V[mask], I[mask]
                G_valid = I_valid / V_valid
                # Filter out outliers using IQR method for smoother envelope
                Q1, Q3 = np.percentile(G_valid, 25), np.percentile(G_valid, 75)
                IQR = Q3 - Q1
                inlier_mask = (G_valid >= Q1 - 1.5 * IQR) & (G_valid <= Q3 + 1.5 * IQR)
                V_clean, G_clean = V_valid[inlier_mask], G_valid[inlier_mask]
                # Group by binned voltage (rounded) to extract max/min conductance per voltage
                df_clean = pd.DataFrame({'V': V_clean, 'G': G_clean})
                df_clean['V_bin'] = np.round(df_clean['V'], 4)
                grouped = df_clean.groupby('V_bin')
                upper = grouped.max().reset_index()
                lower = grouped.min().reset_index()

                # Get up/down sweep colors from original plot
                color_up, color_down = 'blue', 'red'  # default
                for elem in data['elements']:
                        if isinstance(elem, plt.Line2D):
                                label = elem.get_label()
                                if label.startswith("Up-sweep"):
                                        color_up = elem.get_color()
                                elif label.startswith("Down-sweep"):
                                        color_down = elem.get_color()
                # Create a popup window to display the G-V curve
                win = tk.Toplevel()
                win.title(f"G-V Curve: {selected_label}")
                # Create the matplotlib figure and plot upper/lower envelope
                fig_gv, ax_gv = plt.subplots(figsize=(7, 4))
                ax_gv.plot(upper['V_bin'], upper['G'], color=color_up, label='Up-sweep')
                ax_gv.plot(lower['V_bin'], lower['G'], color=color_down, label='Down-sweep')
                # Add zero lines and labels
                ax_gv.axhline(0, color='gray', linestyle='--')
                ax_gv.axvline(0, color='gray', linestyle='--')
                ax_gv.set_xlabel(f"Voltage ({v_to})")
                ax_gv.set_ylabel("Conductance (S)")
                ax_gv.set_title(f"Individual G-V Curve\n{selected_label}")
                ax_gv.legend()
                ax_gv.grid(True, linestyle=':', alpha=0.6)
                fig_gv.tight_layout()
                # Embed plot into the Tkinter window
                canvas = FigureCanvasTkAgg(fig_gv, master=win)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                # Enable interactive cursor (hover to show data points)
                mplcursors.cursor(ax_gv.lines, hover=True).connect(
                        "add", lambda sel: sel.annotation.set_text(
                        f"V = {sel.target[0]:.4g} {v_to}\nG = {sel.target[1]:.4g} S"))
                
                # Add Save Button to export the G-V plot
                def save_gv_plot():
                        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
                        if path:
                                try:
                                        fig_gv.savefig(path, dpi=300, bbox_inches='tight')
                                except Exception as err:
                                        messagebox.showerror("Save Error", f"Failed to save plot:\n{err}")

                tk.Button(win, text="Save G-V Plot", command=save_gv_plot).pack(pady=6)

        # Button to delete the currently selected IV curve
        tk.Button(control_frame, text="Delete Selected IV", command=delete_selected_iv_curve).pack(side=tk.LEFT, padx=5)
        # Button to clear all curves from the plot
        tk.Button(control_frame, text="Clear All", command=clear_all_iv_curves).pack(side=tk.LEFT, padx=5)
        # Button to view the selected IV curve in a popup window
        tk.Button(control_frame, text="View Individual IV", command=view_selected_iv_curve).pack(side=tk.LEFT, padx=5)
        # Button to view the selected GV curve in a popup window
        tk.Button(control_frame, text="View Individual G-V", command=view_selected_gv_curve).pack(side=tk.LEFT, padx=5)
        # Create an additional frame for save/export buttons (can be extended later)
        save_frame = tk.Frame(load_files_and_compute_conductance.plot_frame)
        save_frame.pack(pady=4)
        
        # Function to save the currently plotted IV figure as a PNG image
        def save_plot_interactively():
            # Ask user to choose a file path for saving the plot
            path = filedialog.asksaveasfilename(
                defaultextension=".png", # Default extension if user doesn't provide one
                filetypes=[("PNG Image", "*.png")], # Allow only PNG format
                title="Save IV Plot" # Dialog window title
                )
            if path:
                try:
                    # Save the figure to the chosen path with high resolution and tight layout
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                except Exception as e:
                    # Show error message if saving fails
                    messagebox.showerror("Save Error", f"Failed to save plot:\n{e}")

        # Function to save all plotted IV curve data points to a CSV file
        def save_data_interactively():
            path = filedialog.asksaveasfilename(
                defaultextension=".csv", # Default extension for saving data
                filetypes=[("CSV File", "*.csv")], # Only allow CSV files
                title="Save IV Data" # Dialog window title
                )
            if path:
                try:
                    # Gather all plotted data into one DataFrame
                    rows = []
                    for cid, curve in load_files_and_compute_conductance.curves.items():
                        label = curve["label"] # Get filename/label for this curve
                        for element in curve["elements"]:
                            if isinstance(element, plt.Line2D):
                                V = element.get_xdata()
                                I = element.get_ydata()
                                # Append each (V, I) pair along with curve label
                                for v, i in zip(V, I):
                                    rows.append([label, v, i])
                        # Create DataFrame and save to CSV
                        df_out = pd.DataFrame(rows, columns=["Filename", f"Voltage ({v_to})", f"Current ({i_to})"])
                        df_out.to_csv(path, index=False)
                except Exception as e:
                    # Show error message if data export fails
                    messagebox.showerror("Save Error", f"Failed to save data:\n{e}")
        
        # Functions to save conductance
        def save_conductance_table():
            """
            Save the conductance_table (which stores filename, concentration, and conductance)
            to a CSV file, sorted by concentration in descending order.
            """
            # If the conductance table is empty, notify the user and exit
            if not conductance_table:
                messagebox.showinfo("Info", "No conductance data to save.")
                return

            # Open file dialog for user to choose save location
            file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                     filetypes=[("CSV File", "*.csv")],
                                                     title="Save Conductance Table")
            
            # If a path is selected, proceed to save
            if file_path:
                try:
                    # Create a DataFrame from the conductance_table dictionary, each entry: [filename, concentration, conductance]
                    df_conductance = pd.DataFrame([
                        [os.path.basename(f), conc, G] for f, (conc, G) in conductance_table.items()
                        ], columns=["Filename", "Concentration", "Conductance (S)"])

                    # Convert "Concentration" column to float to ensure numeric sorting. Prevents errors when some values may be strings (from filenames, etc.)
                    df_conductance["Concentration"] = pd.to_numeric(df_conductance["Concentration"], errors='coerce')
                    # Sort by concentration in descending order (from high to low)
                    df_conductance.sort_values(by="Concentration", ascending=False, inplace=True)
                    # Save the table to a CSV file
                    df_conductance.to_csv(file_path, index=False)
                    # Inform the user of success
                    messagebox.showinfo("Saved", f"Conductance table saved to:\n{file_path}")
                
                except Exception as e:
                    # Show an error message if something goes wrong during saving
                    messagebox.showerror("Save Error", f"Failed to save table:\n{e}")
        
        # Unified Save Control Row (Create a frame to hold all save-related buttons in one horizontal row)
        combined_save_frame = tk.Frame(load_files_and_compute_conductance.plot_frame)
        combined_save_frame.pack(pady=6)

        # Left Button：Save Experimental Conductance & Conductivity
        tk.Button(combined_save_frame,
                  text="Save Experimental Conductance and Conductivity",
                  command=save_combined_conductance_conductivity).pack(side=tk.LEFT, padx=5)

        # Right Button：Save Plot / Save Data / Close
        tk.Button(combined_save_frame, text="Save Plot", command=save_plot_interactively).pack(side=tk.LEFT, padx=5)
        tk.Button(combined_save_frame, text="Save Data", command=save_data_interactively).pack(side=tk.LEFT, padx=5)
        #tk.Button(combined_save_frame, text="Close This Plot", command=lambda: load_files_and_compute_conductance.plot_frame.destroy()).pack(side=tk.LEFT, padx=5)
        # Add windows to mathematical show functions, experimental conductivity, use shoelace formula calculate the enclosed area
        tk.Button(combined_save_frame, text="Show Formula", command=lambda: show_formula_popup(
            r'$\sigma = \dfrac{G \cdot \mathrm{length}}{\mathrm{area} \cdot \mathrm{number}}$' + '\n' +
            r'$A = \dfrac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|$' + '\n' +
            r'$A_{\mathrm{norm}} = \dfrac{A - A_{\min}}{A_{\max} - A_{\min}}$',
            title="Formulas for IV Curve Analysis"
            )).pack(side=tk.LEFT, padx=5)

        # Function to remove the legend and all conductance (G) text annotations from the plot
        def hide_legend_and_notation():
                # Remove the legend if it exists
                if hasattr(ax, "legend_") and ax.legend_:
                    ax.legend_.remove()
                # Remove all text annotations from the plot (e.g., "G = ..." texts)
                for txt in list(ax.texts):
                    try:
                        txt.remove()
                    except:
                        pass # Ignore any errors during text removal
                # Redraw the canvas to update the figure
                fig.canvas.draw()

        # Function to restore the legend and re-display conductance (G) annotations
        def show_legend_and_notation():
                # Recreate the legend: include only those lines that have a label and do not start with "_"
                ax.legend(handles=[
                    h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")
                ], loc="upper right") # Position legend in the upper-right corner

                # Clear any existing text annotations to avoid duplicates
                for txt in list(ax.texts):
                    try:
                        txt.remove()
                    except:
                        pass

                # Recalculate and re-display G (conductance) for each curve
                for idx, data in enumerate(load_files_and_compute_conductance.curves.values()):
                    filepath = data["filepath"]

                    # Reload the data file (since slope/G was not stored, we recalculate it)
                    if filepath.endswith(".xlsx"):
                        df = pd.read_excel(filepath)
                    else:
                        df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)
                    
                    # Extract raw voltage (V) and current (I) columns and convert to float arrays
                    V_raw = df.iloc[:, 0].to_numpy(dtype=float)
                    I_raw = df.iloc[:, 1].to_numpy(dtype=float)
                    # Retrieve unit conversion factors for voltage and current
                    v_from = load_files_and_compute_conductance.v_unit_orig.get()
                    v_to = load_files_and_compute_conductance.v_unit
                    i_from = load_files_and_compute_conductance.i_unit_orig.get()
                    i_to = load_files_and_compute_conductance.i_unit
                    # Apply unit conversions to V and I data
                    V = V_raw * (unit_factors[v_from] / unit_factors[v_to])
                    I = I_raw * (unit_factors[i_from] / unit_factors[i_to])
                    # Perform linear regression to compute the slope, which is the conductance (G)
                    slope, _, _, _, _ = linregress(V, I)
                    # Add a text annotation showing the filename and computed G value
                    txt = ax.text(0.02, 0.98 - 0.07 * idx,
                                  f"{os.path.basename(filepath)}\nG = {slope:.2e} S",
                                  transform=ax.transAxes, fontsize=9, color='black',
                                  verticalalignment='top')
                # Redraw the canvas to apply the updated annotations and legend
                fig.canvas.draw()

        # Legend Position Controller
        legend_frame = tk.Frame(load_files_and_compute_conductance.plot_frame) # Create a horizontal frame inside the main plotting frame to hold legend-related controls
        legend_frame.pack(pady=4)# Add vertical padding for spacing
        # Add a label "Legend Position:"
        tk.Label(legend_frame, text="Legend Position:").pack(side=tk.LEFT, padx=5)
        # Create a StringVar to store the selected legend position
        legend_pos_var = tk.StringVar()
        # Create a dropdown (combobox) for legend position selection
        legend_pos_combo = ttk.Combobox(legend_frame, textvariable=legend_pos_var, state="readonly", width=18)
        # Populate the dropdown with available legend locations
        legend_pos_combo['values'] = ['upper right', 'upper left', 'lower right', 'lower left',
                              'center right', 'center left', 'upper center',
                              'lower center', 'center', 'outside lower right'] # 'outside lower right' is a custom case
        # Set the default selection to 'upper right'
        legend_pos_combo.set('upper right')  # Initial position
        legend_pos_combo.pack(side=tk.LEFT, padx=5) # Pack to the left with spacing
        
        # Function to update the legend position on the plot
        def update_legend_position(event):
                # Remove any existing legend on the main axis
                if hasattr(ax, "legend_") and ax.legend_:
                        ax.legend_.remove()
                # Gather the legend handles for all visible labeled lines
                handles = [h for h in ax.lines if h.get_label() and not h.get_label().startswith("_")]
                # Get the selected position from the dropdown
                selected = legend_pos_var.get()
                # Special case: if user chooses 'outside lower right', position legend outside figure
                if selected == "outside lower right":
                        fig.legend(
                                handles=handles,
                                loc='lower right',
                                bbox_to_anchor=(1.0, 0.0),  # Position relative to figure bottom right
                                bbox_transform=fig.transFigure, # Use figure coordinates instead of axis
                                frameon=True # Draw a frame box around the legend
                        )
                else:
                        # Standard legend location within axis
                        ax.legend(handles=handles, loc=selected)
                # Redraw the canvas to update the figure with new legend position
                fig.canvas.draw()

        # Bind the dropdown selection event to the update function
        legend_pos_combo.bind("<<ComboboxSelected>>", update_legend_position)
        # Add a button to hide both legend and conductance annotations
        tk.Button(legend_frame, text="Hide Legend and Notation", command=hide_legend_and_notation).pack(side=tk.LEFT, padx=5)
        # Add a button to restore/show legend and conductance annotations
        tk.Button(legend_frame, text="Show Legend and Notation", command=show_legend_and_notation).pack(side=tk.LEFT, padx=5)

        # Refresh the dropdown list with all curve labels
        curve_selector['values'] = [data["label"] for data in curves.values()]
        curve_selector.set('') # Clear selection after plot

    # If any exception occurs during plotting, show an error message
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot IV curve\n{e}")



'''
def plot_conductance_comparison():
    try:
        df = pd.read_csv("conductance_comparison.csv").dropna().sort_values(by="Concentration_M")
        x = df["Concentration_M"]
        y_exp = df["Experimental_Conductance_S"]
        y_theo = df["Theoretical_Conductance_S"]

        for widget in right_frame.winfo_children():
            widget.destroy()

        plot_frame = tk.Frame(right_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig, ax = plt.subplots(figsize=(6, 5))
        line1, = ax.plot(x, y_theo, 'k-', label="Theoretical conductance")
        line2, = ax.plot(x, y_exp, 'ro', label="Experimental conductance")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Concentration (Mol/L)")
        ax.set_ylabel("Conductance (S)")
        ax.set_title("Conductance vs Concentration")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()

        # 加入交互式标注（悬停查看坐标）
        mplcursors.cursor([line1, line2], hover=True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def close_plot():
            plot_frame.destroy()

        close_btn = tk.Button(plot_frame, text="Close This Plot", command=close_plot)
        close_btn.pack(pady=4)

    except Exception as e:
        messagebox.showerror("Plot Error", f"Failed to plot from conductance_comparison.csv\n{e}")


def merge_and_export_comparison():
    try:
        exp_df = pd.read_csv("conductance_results.csv")
        theo_df = pd.read_csv("theoretical_conductance.csv")

        exp_dict = {extract_concentration_from_filename(row['Filename']): row['Conductance_S'] for _, row in exp_df.iterrows() if extract_concentration_from_filename(row['Filename']) is not None}
        theo_dict = {extract_concentration_from_filename(row['Concentration']): row['Theoretical_Conductance_S'] for _, row in theo_df.iterrows() if extract_concentration_from_filename(row['Concentration']) is not None}

        all_concs = sorted(set(exp_dict) | set(theo_dict))
        data = [[c, exp_dict.get(c, None), theo_dict.get(c, None)] for c in all_concs]
        pd.DataFrame(data, columns=["Concentration_M", "Experimental_Conductance_S", "Theoretical_Conductance_S"]).to_csv("conductance_comparison.csv", index=False)
        messagebox.showinfo("Saved", "Merged conductance_comparison.csv saved successfully")
    except Exception as e:
        messagebox.showerror("Save Error", f"Failed to export comparison CSV\n{e}")

def extract_concentration_from_filename(filename):
    match = re.search(r"([\d\.Ee\-]+)\s*M", filename)
    return float(match.group(1)) if match else None
'''

'''
import matplotlib.pyplot as plt

def plot_g_vs_v():
    for widget in right_frame.winfo_children():
        widget.destroy()

    try:
        filepaths = filedialog.askopenfilenames(filetypes=[("Excel and Text files", "*.xlsx *.txt")])
        if not filepaths:
            return

        for filepath in filepaths:
            # 读取数据
            if filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)

            V = df.iloc[:, 0].to_numpy(dtype=float)
            I = df.iloc[:, 1].to_numpy(dtype=float)

            # 去除 V = 0 防止除零错误
            mask = ~np.isclose(V, 0)
            V_valid, I_valid = V[mask], I[mask]
            G_valid = I_valid / V_valid

            # 去除异常值（IQR）
            Q1, Q3 = np.percentile(G_valid, 25), np.percentile(G_valid, 75)
            IQR = Q3 - Q1
            inlier_mask = (G_valid >= Q1 - 1.5 * IQR) & (G_valid <= Q3 + 1.5 * IQR)
            V_clean, G_clean = V_valid[inlier_mask], G_valid[inlier_mask]

            # 整理包络线
            df_clean = pd.DataFrame({'V': V_clean, 'G': G_clean})
            df_clean['V_bin'] = np.round(df_clean['V'], 4)

            grouped = df_clean.groupby('V_bin')
            upper_envelope = grouped.max().reset_index()
            lower_envelope = grouped.min().reset_index()

            # 创建画布
            if hasattr(plot_g_vs_v, "plot_frame") and plot_g_vs_v.plot_frame.winfo_exists():
                plot_g_vs_v.plot_frame.destroy()

            plot_g_vs_v.plot_frame = tk.Frame(right_frame)
            plot_g_vs_v.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

            fig, ax = plt.subplots(figsize=(8, 6))

            # 绘制包络线
            line_upper = ax.plot(upper_envelope['V_bin'], upper_envelope['G'],
                                 color='blue', label='Up-sweep')[0]
            line_lower = ax.plot(lower_envelope['V_bin'], lower_envelope['G'],
                                 color='red', label='Down-sweep')[0]

            # 坐标轴样式
            ax.axhline(0, color='gray', linestyle='--')
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Conductance (S)")
            ax.set_title(f"Experimental Conductance vs Voltage\n{os.path.basename(filepath)}")
            ax.legend()
            ax.grid(True)
            fig.tight_layout()

            # 悬停交互功能
            mplcursors.cursor([line_upper, line_lower], hover=True).connect(
                "add", lambda sel: sel.annotation.set_text(
                    f"V = {sel.target[0]:.4g} V\nG = {sel.target[1]:.4g} S"
                )
            )

            canvas = FigureCanvasTkAgg(fig, master=plot_g_vs_v.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 添加导出图像/数据等功能
            add_plot_utilities(fig, ax, plot_g_vs_v.plot_frame,
                               upper_envelope['V_bin'], upper_envelope['G'],
                               x_label="Voltage (V)", y_label="Conductance (S)",
                               filename_prefix="G_vs_V")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot G-V for {filepath}\n{e}")

'''
# %% [8.5] Main Function 3.5: Concentration and Conductance (Conductivity) Analysis    
def choose_and_merge_conductance_tables():
    try:
        # ✅ 提示用户选择实验数据表
        messagebox.showinfo("Instruction", "Choose experimental conductance table")
        exp_path = filedialog.askopenfilename(title="Select Experimental Conductance CSV",
                                              filetypes=[("CSV files", "*.csv")])
        if not exp_path:
            return

        # ✅ 提示用户选择理论数据表
        messagebox.showinfo("Instruction", "Choose theoretical conductance table")
        theo_path = filedialog.askopenfilename(title="Select Theoretical Conductance CSV",
                                               filetypes=[("CSV files", "*.csv")])
        if not theo_path:
            return

        # 读取文件
        exp_df = pd.read_csv(exp_path)
        theo_df = pd.read_csv(theo_path)

        # 自动识别列名
        exp_conc_col = next((col for col in exp_df.columns if "Concentration" in col), None)
        theo_conc_col = next((col for col in theo_df.columns if "Concentration" in col), None)
        exp_cond_col = next((col for col in exp_df.columns if "Conductance" in col), None)
        theo_cond_col = next((col for col in theo_df.columns if "Theoretical" in col), None)

        if not (exp_conc_col and theo_conc_col and exp_cond_col and theo_cond_col):
            raise KeyError("Missing required column in one of the tables.")

        # 清洗数值
        exp_df["Concentration"] = pd.to_numeric(exp_df[exp_conc_col], errors='coerce')
        theo_df["Concentration"] = pd.to_numeric(theo_df[theo_conc_col], errors='coerce')

        # 构建合并表
        df_combined = pd.merge(
            exp_df[["Concentration", exp_cond_col]].rename(columns={exp_cond_col: "Experimental_Conductance_S"}),
            theo_df[["Concentration", theo_cond_col]].rename(columns={theo_cond_col: "Theoretical_Conductance_S"}),
            on="Concentration", how="outer"
        )

        df_combined.sort_values(by="Concentration", ascending=False, inplace=True)
        df_combined.rename(columns={"Concentration": "Concentration_M"}, inplace=True)
        df_combined.to_csv("conductance_comparison.csv", index=False)

        show_combined_table(df_combined)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to merge tables:\n{e}")


def show_combined_table(df):
    win = tk.Toplevel()
    win.title("Merged Conductance Table")
    win.geometry("600x400")

    frame = tk.Frame(win)
    frame.pack(fill=tk.BOTH, expand=True)

    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show="headings")
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=200)
    for _, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))
    tree.pack(fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    tk.Button(win, text="Close", command=win.destroy).pack(pady=5)

def plot_conductance_comparison():
    try:
        df = pd.read_csv("conductance_comparison.csv").dropna().sort_values(by="Concentration_M", ascending=False)
        x = df["Concentration_M"]
        y_exp = df["Experimental_Conductance_S"]
        y_theo = df["Theoretical_Conductance_S"]

        for widget in right_frame.winfo_children():
            widget.destroy()

        plot_frame = tk.Frame(right_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig, ax = plt.subplots(figsize=(6, 5))
        line1, = ax.plot(x, y_theo, 'k-', label="Theoretical conductance")
        line2, = ax.plot(x, y_exp, 'ro', label="Experimental conductance")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Concentration (Mol/L)")
        ax.set_ylabel("Conductance (S)")
        ax.set_title("Conductance vs Concentration")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()

        mplcursors.cursor([line1, line2], hover=True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        add_plot_utilities(
            fig=fig,
            ax=ax,
            plot_frame=plot_frame,
            x=x,
            y=y_exp,
            x_label="Concentration (Mol/L)",
            y_label="Experimental Conductance (S)",
            filename_prefix="conductance_comparison"
            )

    except Exception as e:
        messagebox.showerror("Plot Error", f"Failed to plot from conductance_comparison.csv\n{e}")

def open_analysis_ui():
    for widget in right_frame.winfo_children():
        widget.destroy()

    # ------------------ Conductance ------------------
    tk.Label(right_frame, text="Conductance Analysis", font=("Arial", 12, "bold")).pack(pady=5)

    combine_btn = tk.Button(right_frame, text="Choose Experimental and Theoretical Conductance Tables to Combine",
                            command=choose_and_merge_conductance_tables)
    combine_btn.pack(pady=3)

    plot_btn = tk.Button(right_frame, text="Plot Conductance Comparison",
                         command=plot_conductance_comparison)
    plot_btn.pack(pady=3)

    # ------------------ Conductivity ------------------
    tk.Label(right_frame, text="Conductivity Analysis", font=("Arial", 12, "bold")).pack(pady=(15, 5))

    combine_sigma_btn = tk.Button(right_frame, text="Choose Experimental and Theoretical Conductivity Tables to Combine",
                                  command=choose_and_merge_conductivity_tables)
    combine_sigma_btn.pack(pady=3)

    plot_sigma_btn = tk.Button(right_frame, text="Plot Conductivity Comparison",
                               command=plot_conductivity_comparison)
    plot_sigma_btn.pack(pady=3)

def open_conductivity_analysis_ui():
    for widget in right_frame.winfo_children():
        widget.destroy()

    combine_btn = tk.Button(right_frame, text="Choose Experimental and Theoretical Conductivity Tables to Combine",
                            command=choose_and_merge_conductivity_tables)
    combine_btn.pack(pady=10)

    plot_btn = tk.Button(right_frame, text="Plot Conductivity Comparison",
                         command=plot_conductivity_comparison)
    plot_btn.pack(pady=5)

def choose_and_merge_conductivity_tables():
    try:
        messagebox.showinfo("Instruction", "Choose experimental conductivity table")
        exp_path = filedialog.askopenfilename(title="Select Experimental Conductivity CSV",
                                              filetypes=[("CSV files", "*.csv")])
        if not exp_path:
            return

        messagebox.showinfo("Instruction", "Choose theoretical conductivity table")
        theo_path = filedialog.askopenfilename(title="Select Theoretical Conductivity CSV",
                                               filetypes=[("CSV files", "*.csv")])
        if not theo_path:
            return

        exp_df = pd.read_csv(exp_path)
        theo_df = pd.read_csv(theo_path)

        exp_conc_col = next((col for col in exp_df.columns if "Concentration" in col), None)
        theo_conc_col = next((col for col in theo_df.columns if "Concentration" in col), None)
        exp_sigma_col = next((col for col in exp_df.columns if "Conductivity" in col), None)
        theo_sigma_col = next((col for col in theo_df.columns if "Theoretical" in col), None)

        if not (exp_conc_col and theo_conc_col and exp_sigma_col and theo_sigma_col):
            raise KeyError("Missing required column in one of the tables.")

        exp_df["Concentration"] = pd.to_numeric(exp_df[exp_conc_col], errors='coerce')
        theo_df["Concentration"] = pd.to_numeric(theo_df[theo_conc_col], errors='coerce')

        df_combined = pd.merge(
            exp_df[["Concentration", exp_sigma_col]].rename(columns={exp_sigma_col: "Experimental_Conductivity_S/m"}),
            theo_df[["Concentration", theo_sigma_col]].rename(columns={theo_sigma_col: "Theoretical_Conductivity_S/m"}),
            on="Concentration", how="outer"
        )

        df_combined.sort_values(by="Concentration", ascending=False, inplace=True)
        df_combined.rename(columns={"Concentration": "Concentration_M"}, inplace=True)
        df_combined.to_csv("conductivity_comparison.csv", index=False)

        show_combined_table(df_combined)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to merge tables:\n{e}")

def plot_conductivity_comparison():
    try:
        df = pd.read_csv("conductivity_comparison.csv").dropna().sort_values(by="Concentration_M", ascending=False)
        x = df["Concentration_M"]
        y_exp = df["Experimental_Conductivity_S/m"]
        y_theo = df["Theoretical_Conductivity_S/m"]

        for widget in right_frame.winfo_children():
            widget.destroy()

        plot_frame = tk.Frame(right_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        fig, ax = plt.subplots(figsize=(6, 5))
        line1, = ax.plot(x, y_theo, 'k-', label="Theoretical conductivity")
        line2, = ax.plot(x, y_exp, 'ro', label="Experimental conductivity")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Concentration (Mol/L)")
        ax.set_ylabel("Conductivity (S/m)")
        ax.set_title("Conductivity vs Concentration")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        fig.tight_layout()

        mplcursors.cursor([line1, line2], hover=True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        add_plot_utilities(
            fig=fig,
            ax=ax,
            plot_frame=plot_frame,
            x=x,
            y=y_exp,
            x_label="Concentration (Mol/L)",
            y_label="Experimental Conductivity (S/m)",
            filename_prefix="conductivity_comparison"
        )

    except Exception as e:
        messagebox.showerror("Plot Error", f"Failed to plot from conductivity_comparison.csv\n{e}")
# %% [9] Main Function 4: Drift-Diffusion Analysis
# %%% [9.1] Helper Function: Unit Coverter
# Convert genral unit of conductivity function
def convert_units(value_entry, conc_entry, from_unit, to_unit, result_label):
    """
    Convert a conductivity value between different units, including both bulk conductivity
    (e.g., S/m, mS/cm) and molar conductivity units (e.g., S·cm²/mol, S·m²/mol).
    
    This function also uses the solution concentration (in mol/L) when converting to or from
    molar conductivity units, since:
        λ (molar conductivity) = σ / c
        σ (bulk conductivity) = λ × c

    Parameters:
        value_entry: Entry widget containing the input conductivity value
        conc_entry: Entry widget containing the solution concentration (mol/L)
        from_unit: Combobox widget with the selected input unit
        to_unit: Combobox widget with the selected output unit
        result_label: Entry widget where the converted value will be displayed
    """
    try:
        val = float(value_entry.get()) # Original conductivity value
        conc = float(conc_entry.get()) # Concentration in mol/L
        original = from_unit.get()     # Original unit string
        target = to_unit.get()         # Target unit string

        # If source and target units are the same, return original value
        if original == target:
            result_label.delete(0, tk.END)
            result_label.insert(0, f"{val:.4g} {target}")
            return

        # Convert all source units to base SI unit: S/m
        if original == "mS/cm":
            val_s_per_m = val * 0.1                   # 1 mS/cm = 0.1 S/m
        elif original == "μS/cm":
            val_s_per_m = val * 1e-4                  # 1 μS/cm = 0.0001 S/m
        elif original == "mS/m":
            val_s_per_m = val * 0.001                 # 1 mS/m = 0.001 S/m
        elif original == "μS/m":
            val_s_per_m = val * 1e-6                  # 1 μS/m = 0.000001 S/m
        elif original == "S·cm²/mol":
            val_s_per_m = val * 1e-4 * conc * 1000
        elif original == "S·m²/mol":
            val_s_per_m = val * conc * 1000
        elif original == "S/m":
            val_s_per_m = val
        else:
            raise ValueError("Unsupported original unit")

        if target == "mS/cm":
            result = val_s_per_m / 0.1
        elif target == "μS/cm":
            result = val_s_per_m / 1e-4
        elif target == "mS/m":
            result = val_s_per_m / 1e-3
        elif target == "μS/m":
            result = val_s_per_m / 1e-6
        elif target == "S·cm²/mol":
            result = val_s_per_m / (conc * 1000 * 1e-4)
        elif target == "S·m²/mol":
            result = val_s_per_m / (conc * 1000)
        elif target == "S/m":
            result = val_s_per_m
        else:
            raise ValueError("Unsupported target unit")

        result_label.delete(0, tk.END)
        result_label.insert(0, f"{result:.4g} {target}")
    except Exception as e:
        messagebox.showerror("Conversion Error", str(e))

# %%% [9.2] Helper Function: Power Density Calculation
def calculate_power_density_from_table():
    global nanochannel_dimensions
    try:
        # Read geometry: from saved or entry
        if use_saved_dims_var.get():
            h = float(nanochannel_dimensions["height"])
            w = float(nanochannel_dimensions["width"])
            n = int(nanochannel_dimensions["number"])
        else:
            h = float(height_entry.get())
            w = float(width_entry.get())
            n = int(num_channels_entry.get())
            
            # ✅ 更新保存值
            nanochannel_dimensions["height"] = h
            nanochannel_dimensions["width"] = w
            nanochannel_dimensions["number"] = n

        # ✅ 从用户输入读取 Em 和 G
        Em = float(em_entry.get())
        G = float(conductance_entry.get())

        P_osm = 0.25 * Em**2 * G
        A_eff = n * h * w
        P_density = P_osm / A_eff

        messagebox.showinfo("Power Density", f"Max Power Density:\n{P_density:.4e} W/m²")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to calculate:\n{e}")

def calc_conductance(V: np.ndarray,
                     I: np.ndarray,
                     v_min: float | None = None,
                     v_max: float | None = None) -> float:
    """Return conductance G (S).

    If *v_min* and *v_max* are both provided, only fit the data where
    v_minÂ â‰¤Â VÂ â‰¤Â v_max. Otherwise, fit the whole curve.
    """
    if v_min is not None and v_max is not None:
        mask = (V >= min(v_min, v_max)) & (V <= max(v_min, v_max))
        if mask.sum() < 2:
            raise ValueError("Not enough points in selected range.")
        V_fit, I_fit = V[mask], I[mask]
    else:
        V_fit, I_fit = V, I

    slope, _, _, _, _ = linregress(V_fit, I_fit)
    return slope   # [S]

# %%% [9.3] Core Function: MAIN Function of Drift - Diffusion Experiment
def run_drift_diffusion_experiment(root, right_frame):
    global drift_diffusion_results, drift_diffusion_table, ion_input_entries, tree
    for widget in right_frame.winfo_children():
        widget.destroy()

    def compute_monovalent_redox():
        gamma_H = float(ion_input_entries["γ_H (High)"].get())
        gamma_L = float(ion_input_entries["γ_L (Low)"].get())
        C_H = float(ion_input_entries["C_H (mol/L)"].get())
        C_L = float(ion_input_entries["C_L (mol/L)"].get())
        return (k_B * T / e) * np.log((gamma_H * C_H) / (gamma_L * C_L))
    def add_blank_row():
        values = ["", "", "", "", "", "", ""]
        tree.insert("", tk.END, values=values)

    def update_single_row():
        try:
            selected = tree.selection()
            if not selected:
                messagebox.showinfo("No Selection", "Please select a row to update.")
                return

            for row in selected:
                redox_str = tree.item(row)['values'][2]
                em_str = tree.item(row)['values'][3]
                redox = None  # 🟢 初始化 redox 避免未赋值

                if em_str.strip():
                    Em = float(em_str)
                else:
                    try:
                        tot = float(tree.item(row)['values'][1])
                    except:
                        messagebox.showerror("Missing Total Potential", "Please input Total Potential or Em.")
                        return

                    if redox_str.strip() == "":
                        try:
                            redox = compute_monovalent_redox()
                        except:
                            messagebox.showerror("Input Error", "Please input Redox Potential in the table or fill all γ_H, γ_L, C_H, C_L.")
                            return
                    else:
                        try:
                            redox = float(redox_str)
                        except:
                            messagebox.showerror("Format Error", "Redox Potential must be a number.")
                            return

                    Em = tot - redox

                try:
                    C_H = float(ion_input_entries["C_H (mol/L)"].get())
                    C_L = float(ion_input_entries["C_L (mol/L)"].get())
                except:
                    messagebox.showerror("Missing C_H or C_L", "Please input both C_H and C_L for mobility calculation.")
                    return

                delta_C = C_H / C_L

                try:
                    sigma = float(ion_input_entries["σ (S/m)"].get())
                except:
                    messagebox.showerror("Missing σ", "Please input conductivity σ (S/m).")
                    return

                numerator = np.log(delta_C) + e * Em / (k_B * T)
                denominator = np.log(delta_C) - e * Em / (k_B * T)

                if abs(denominator) < 1e-10:
                    messagebox.showerror("Math Error", "Denominator too small. μ⁺/μ⁻ ratio approaches infinity.")
                    return

                ratio = numerator / denominator

                if abs(1 + ratio) < 1e-8:
                    messagebox.showerror("Math Error", "μ⁺/μ⁻ = -1 → μ⁻ = ∞. Please adjust Redox or Concentrations.")
                    return

                mu_sum = sigma / (e * N_A * C_H * 1000)  # C in mol/L
                mu_minus = mu_sum / (1 + ratio)
                mu_plus = mu_sum - mu_minus

                # Only append the table when the redox is given some value
                if redox is not None:
                    tree.set(row, "Redox Potential", f"{redox:.8f}")
                tree.set(row, "Em", f"{Em:.4g}")
                tree.set(row, "μ⁺/μ⁻", f"{ratio:.4g}")
                tree.set(row, "μ⁺", f"{mu_plus:.4g}")
                tree.set(row, "μ⁻", f"{mu_minus:.4g}")

        except Exception as err:
            messagebox.showerror("Error", f"Update failed: {err}")


    def compute_nonmonovalent():
        try:
            z_plus = float(ion_input_entries["z⁺"].get())
            z_minus = float(ion_input_entries["z⁻"].get())
            C_H = float(ion_input_entries["C_H_general (mol/L)"].get())
            C_L = float(ion_input_entries["C_L_general (mol/L)"].get())
            C_plus = float(ion_input_entries["C⁺ (mol/L)"].get())
            C_minus = float(ion_input_entries["C⁻ (mol/L)"].get())
            sigma = float(ion_input_entries["σ_general (S/m)"].get())

            selected = tree.selection()
            if not selected:
                messagebox.showinfo("No Selection", "Please select a row in the table.")
                return

            redox = None  # Redox initialisation
            em_str = tree.item(selected[0])['values'][3]
            if em_str.strip():
                E_osm = float(em_str)
            else:
                try:
                    E_tot = float(tree.item(selected[0])['values'][1])
                except:
                    messagebox.showerror("Missing Total Potential", "Please input Total Potential or Em.")
                    return

                try:
                    redox = float(ion_input_entries["Redox_general (V)"].get())
                    redox = round(redox, 8)
                except:
                    messagebox.showerror("Missing Redox", "Please input Redox_general (V) or directly input Em.")
                    return

                E_osm = E_tot - redox

            ln_delta = np.log(C_H / C_L)

            numerator = ln_delta - (z_minus * F * E_osm) / (R * T)
            denominator = ln_delta - (z_plus * F * E_osm) / (R * T)

            if abs(denominator) < 1e-10:
                messagebox.showerror("Math Error", "Denominator too small. μ⁺/μ⁻ diverges.")
                return

            alpha = abs((z_plus / z_minus) * (-numerator / denominator))  # μ⁺ / μ⁻

            denom_mu_minus = F * (abs(z_plus) * alpha * C_plus + abs(z_minus) * C_minus)
            if abs(denom_mu_minus) < 1e-10:
                messagebox.showerror("Math Error", "Conductivity formula denominator too small.")
                return

            mu_minus = (sigma / denom_mu_minus) / 1000
            mu_plus = alpha * mu_minus

            if redox is not None:
                tree.set(selected[0], "Redox Potential", f"{redox:.8f}")
            tree.set(selected[0], "Em", f"{E_osm:.4g}")
            tree.set(selected[0], "μ⁺/μ⁻", f"{alpha:.4g}")
            tree.set(selected[0], "μ⁺", f"{mu_plus:.4g}")
            tree.set(selected[0], "μ⁻", f"{mu_minus:.4g}")

        except Exception as err:
            messagebox.showerror("Error", f"Failed to compute non-monovalent mobility:\n{err}")


    def save_table_to_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not file_path:
            return
        df_out = pd.DataFrame([tree.item(row)['values'] for row in tree.get_children()], columns=cols)
        df_out.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Table saved to:\n{file_path}")

    def clear_table():
        for row in tree.get_children():
            tree.delete(row)
        drift_diffusion_results.clear()

    def delete_selected_row():
        selected = tree.selection()
        if not selected:
            return
        for item in selected:
            index = tree.index(item)
            tree.delete(item)
            del drift_diffusion_results[index]

    def upload_and_plot_iv():
        filepath = filedialog.askopenfilename(filetypes=[("Excel and Text files", "*.xlsx *.txt")])
        if not filepath:
            return
        try:
            if filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, header=None)
            V = df.iloc[:, 0].to_numpy(dtype=float)
            I = df.iloc[:, 1].to_numpy(dtype=float)
            cross_V = []
            for i in range(len(I) - 1):
                if I[i] * I[i + 1] < 0:
                    v0, v1 = V[i], V[i + 1]
                    i0, i1 = I[i], I[i + 1]
                    v_cross = v0 - i0 * (v1 - v0) / (i1 - i0)
                    cross_V.append(v_cross)
            
            if not cross_V:
                messagebox.showinfo("No Crossing", "No I=0 crossing point found.")
                return
            
            cross_V = sorted(cross_V)
            etotal_left = cross_V[0]
            etotal_right = cross_V[-1]
            etotal_mid = np.mean(cross_V)

            v_mid = cross_V[0] if len(cross_V) == 1 else np.mean(cross_V)
            
            # ---------- 全曲线电导 ----------
            conductance = calc_conductance(V, I)
            
            conc = os.path.splitext(os.path.basename(filepath))[0]
            values = [conc, f"{v_mid:.4g}"] + ["" for _ in range(3)]
            tree.insert("", tk.END, values=values)
            drift_diffusion_results.append([conc, v_mid, conductance])
            
            # ---------- 绘图 ----------
            fig_win = tk.Toplevel()
            fig, ax = plt.subplots()

            # === 创建 Canvas BEFORE draw ===
            canvas = FigureCanvasTkAgg(fig, fig_win)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # === 绘制所有曲线 ===
            ax.plot(V, I, label="IV Curve", linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--")
            ax.axvline(0, color="gray", linestyle="--")
            ax.plot(v_mid, 0, "o", color="gold", label="Zero-current point (Etotal)")
            ax.plot(etotal_left, 0, 'o', color='red', label="Left Etotal")
            ax.plot(etotal_mid, 0, 'o', color='gold', label="Mid Etotal")
            ax.plot(etotal_right, 0, 'o', color='blue', label="Right Etotal")

            # === 图像细节 ===
            ax.set_xlabel("Voltage (V)", fontsize=12)
            ax.set_ylabel("Current (A)", fontsize=12)
            ax.set_title("IV Curve with Em and Conductance", fontsize=13)
            ax.tick_params(labelsize=10)
            ax.grid(True)

            # ✅ 统一调用 legend()
            ax.legend(fontsize=10)

            # ✅ 最后绘制
            canvas.draw()

            # === Etotal selection ===
            etotal_choice_var = tk.StringVar()
            etotal_choice_var.set("Mid")  # default
            
            def update_etotal_selection(*args):
                selected = etotal_choice_var.get()
                if selected == "Left":
                    v_mid = etotal_left
                elif selected == "Right":
                    v_mid = etotal_right
                else:
                    v_mid = etotal_mid
                    # 更新表格 & drift_diffusion_results
                conc = os.path.splitext(os.path.basename(filepath))[0]
                values = [conc, f"{v_mid:.4g}"] + ["" for _ in range(5)]
                tree.insert("", tk.END, values=values)
                drift_diffusion_results.append([conc, v_mid])
                    
                    # 放置选择框
            etotal_frame = tk.Frame(fig_win)
            etotal_frame.pack(pady=5)
            
            tk.Label(etotal_frame, text="Select Etotal to Use:").pack(side=tk.LEFT)
            etotal_options = ["Left", "Mid", "Right"]
            etotal_menu = tk.OptionMenu(etotal_frame, etotal_choice_var, *etotal_options, command=lambda _: update_etotal_selection())
            etotal_menu.pack(side=tk.LEFT)


            mplcursors.cursor(ax, hover=True)
            # 在这里添加 entry 框
            entry_frame = tk.Frame(fig_win)
            entry_frame.pack(pady=5)
            
            tk.Label(entry_frame, text="Full-range Conductance (S):").pack(side=tk.LEFT)
            
            conductance_entry = tk.Entry(entry_frame, width=25)
            conductance_entry.insert(0, f"{conductance:.4e}")
            conductance_entry.pack(side=tk.LEFT)
            # === 添加 Zoom 控制 ===
            zoom_frame = tk.Frame(fig_win)
            zoom_frame.pack(pady=5)
            attach_zoom_controls(zoom_frame, fig, ax)

        
            def calculate_selected_region():
                region_win = tk.Toplevel()
                region_win.title("Selected Region Conductance")

                # === 上方：图像区域 ===
                fig, ax = plt.subplots()
                canvas = FigureCanvasTkAgg(fig, master=region_win)
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                ax.set_xlabel("Voltage (V)", fontsize=12)
                ax.set_ylabel("Current (A)", fontsize=12)
                ax.set_title("Selected Region IV and Conductance", fontsize=13)
                ax.grid(True)
                ax.tick_params(labelsize=10)
                ax.axhline(0, color="gray", linestyle="--")
                ax.axvline(0, color="gray", linestyle="--")

                # === 下方：输入与结果 ===
                input_frame = tk.Frame(region_win)
                input_frame.pack(pady=5)

                tk.Label(input_frame, text="Start Voltage (V):").grid(row=0, column=0, padx=5, pady=2)
                v1_entry = tk.Entry(input_frame, width=12)
                v1_entry.grid(row=0, column=1, padx=5, pady=2)

                tk.Label(input_frame, text="End Voltage (V):").grid(row=1, column=0, padx=5, pady=2)
                v2_entry = tk.Entry(input_frame, width=12)
                v2_entry.grid(row=1, column=1, padx=5, pady=2)

                result_var = tk.StringVar()
                tk.Label(input_frame, textvariable=result_var, fg="blue").grid(row=2, column=0, columnspan=2, pady=5)

                tk.Label(input_frame, text="Local G (S):").grid(row=3, column=0, padx=5, pady=2)
                g_entry = tk.Entry(input_frame, width=20)
                g_entry.grid(row=3, column=1, padx=5, pady=2)

                def compute_and_plot():
                    try:
                        v1 = float(v1_entry.get())
                        v2 = float(v2_entry.get())
                        idx = np.where((V >= min(v1, v2)) & (V <= max(v1, v2)))[0]
                        if len(idx) < 2:
                            result_var.set("Not enough points.")
                            return
                        subV, subI = V[idx], I[idx]
                        local_G = calc_conductance(V, I, v1, v2)
                        result_var.set(f"G = {local_G:.2e} S")
                        g_entry.delete(0, tk.END)
                        g_entry.insert(0, f"{local_G:.4e}")

                        # 更新图像
                        ax.clear()
                        ax.plot(subV, subI, label="Selected IV", linewidth=1.5)
                        ax.axhline(0, color="gray", linestyle="--")
                        ax.axvline(0, color="gray", linestyle="--")
                        ax.set_xlabel("Voltage (V)", fontsize=12)
                        ax.set_ylabel("Current (A)", fontsize=12)
                        ax.set_title("Selected Region IV and Conductance", fontsize=13)
                        ax.grid(True)
                        ax.tick_params(labelsize=10)
                        ax.legend(fontsize=10)



                        canvas.draw()
                        mplcursors.cursor(ax, hover=True)

                    except ValueError:
                        result_var.set("Invalid input.")

                tk.Button(input_frame, text="Calculate", command=compute_and_plot).grid(
                    row=4, column=0, columnspan=2, pady=8
                )



            tk.Button(
                fig_win,
                text="Calculate Conductance in Selected Region",
                command=calculate_selected_region
                ).pack(pady=5)

            # ---------- Save last curve globally ----------
            global last_uploaded_iv
            last_uploaded_iv = {
                "V": V,
                "I": I,
                "filename": conc,
                "ax": ax,
                "figure": fig
                }

        except Exception as e:
            messagebox.showerror("Error", str(e))


    drift_diffusion_table = tk.Frame(right_frame)
    drift_diffusion_table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    top_frame = tk.Frame(drift_diffusion_table)
    top_frame.pack(fill=tk.X)

    # Layout form：Monovalent on the left ，Non-monovalent on the right
    mono_frame = tk.Frame(top_frame)
    mono_frame.pack(side=tk.LEFT, padx=(0, 20), anchor="n")

    nonmono_frame = tk.Frame(top_frame)
    nonmono_frame.pack(side=tk.LEFT, anchor="n")

    tk.Label(mono_frame, text="Monovalent Salt", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

    input_fields = ["γ_H (High)", "γ_L (Low)", "C_H (mol/L)", "C_L (mol/L)", "σ (S/m)"]
    for field in input_fields:
        row = tk.Frame(mono_frame)
        tk.Label(row, text=field, width=15).pack(side=tk.LEFT)
        entry = tk.Entry(row, width=15)
        entry.pack(side=tk.LEFT)
        row.pack(anchor="w", pady=1)
        ion_input_entries[field] = entry
    
    # Frame contains Redox button and display
    redox_row = tk.Frame(mono_frame)
    redox_row.pack(anchor="w", pady=(8, 2))

    tk.Button(redox_row, text="Calculate Redox Potential", command=lambda: calculate_redox()).pack(side=tk.LEFT, padx=(0, 5))

    redox_display = tk.Entry(redox_row, width=15)
    redox_display.pack(side=tk.LEFT)

    def calculate_redox():
        try:
            redox = compute_monovalent_redox()
            redox_display.delete(0, tk.END)
            redox_display.insert(0, f"{redox:.4f}")
        except Exception as err:
            messagebox.showerror("Error", f"Failed to calculate Redox Potential:\n{err}")

    tk.Label(nonmono_frame, text="General Drift-Diffusion Formula", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

    fields_nonmono = [
    "z⁺", "z⁻",
    "C_H_general (mol/L)", "C_L_general (mol/L)",
    "C⁺ (mol/L)", "C⁻ (mol/L)",
    "σ_general (S/m)", "Redox_general (V)"
    ]


    for field in fields_nonmono:
        row = tk.Frame(nonmono_frame)
        tk.Label(row, text=field, width=15).pack(side=tk.LEFT)
        entry = tk.Entry(row, width=15)
        entry.pack(side=tk.LEFT)
        row.pack(anchor="w", pady=1)
        ion_input_entries[field] = entry
    # Enter salt name (eg. KCl)
    salt_row = tk.Frame(nonmono_frame)
    salt_row.pack(anchor="w", pady=(10, 2))

    tk.Label(salt_row, text="Salt (e.g., Mg1Cl2):", width=20).pack(side=tk.LEFT)
    salt_entry = tk.Entry(salt_row, width=18)
    salt_entry.pack(side=tk.LEFT, padx=5)
    tk.Button(salt_row, text="Apply", command=lambda: apply_salt()).pack(side=tk.LEFT)

    def apply_salt():
        try:
            import re
            formula = salt_entry.get().strip()
            if not formula:
                messagebox.showerror("Input Error", "Please enter a salt formula like K1Cl1 or Mg1Cl2.")
                return

            matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
            if len(matches) != 2:
                messagebox.showerror("Format Error", "Only binary salts like K1Cl1 or MgCl2 are supported.")
                return

            (cation, n1), (anion, n2) = matches
            n1 = int(n1) if n1 else 1
            n2 = int(n2) if n2 else 1

            # Fundamental data set (Can be extended)
            charge_dict = {
                'K': 1, 'Na': 1, 'Li': 1, 'Ag': 1,
                'Mg': 2, 'Ca': 2, 'Ba': 2, 'Zn': 2,
                'Al': 3, 'Fe': 3,
                'Cl': -1, 'Br': -1, 'I': -1, 'F': -1,
                'SO4': -2, 'NO3': -1, 'PO4': -3,
            }

            z_plus = charge_dict.get(cation, None)
            z_minus = charge_dict.get(anion, None)

            if z_plus is None or z_minus is None:
                messagebox.showerror("Charge Error", f"Unknown ion: {cation} or {anion}. Extend charge_dict if needed.")
                return

            z_plus = abs(z_plus)
            z_minus = abs(z_minus)

            # 判断总正负电数是否平衡
            if n1 * z_plus != n2 * z_minus:
                messagebox.showwarning("Charge Imbalance",
                    f"Warning: {n1}{cation}^{z_plus}+ and {n2}{anion}^{z_minus}- are not neutral. Proceeding anyway.")

            # 从 C_H_general 获取总浓度
            try:
                C_H_val = float(ion_input_entries["C_H_general (mol/L)"].get())
            except:
                messagebox.showerror("Missing C_H", "Please input C_H_general (mol/L) first.")
                return

            # molar ratio 分配
            C_cation = n1 * C_H_val
            C_anion = n2 * C_H_val


            # 设置输入字段
            ion_input_entries["z⁺"].delete(0, tk.END)
            ion_input_entries["z⁺"].insert(0, str(z_plus))
            ion_input_entries["z⁻"].delete(0, tk.END)
            ion_input_entries["z⁻"].insert(0, str(-z_minus))  # 注意负号
            ion_input_entries["C⁺ (mol/L)"].delete(0, tk.END)
            ion_input_entries["C⁺ (mol/L)"].insert(0, f"{C_cation:.4g}")
            ion_input_entries["C⁻ (mol/L)"].delete(0, tk.END)
            ion_input_entries["C⁻ (mol/L)"].insert(0, f"{C_anion:.4g}")

        except Exception as err:
            messagebox.showerror("Error", f"Failed to apply salt: {err}")

    

    upload_btn_frame = tk.Frame(drift_diffusion_table)
    upload_btn_frame.pack(pady=(5, 15))
    
    # 新建子容器用于水平排列两个按钮
    btn_row = tk.Frame(upload_btn_frame)
    btn_row.pack()
    
    # 左侧按钮：Upload IV
    tk.Button(btn_row, text="Upload IV", command=upload_and_plot_iv).pack(side=tk.LEFT, padx=10)
    
    # 右侧按钮：Add Row (No IV)
    tk.Button(btn_row, text="Add Row (No IV)", command=add_blank_row).pack(side=tk.RIGHT, padx=10)


    conversion_frame = tk.Frame(top_frame)
    conversion_frame.pack(side=tk.RIGHT, anchor="n", padx=(40, 0))
    
    # === Power Density Calculation Panel ===
    power_frame = tk.Frame(top_frame)
    power_frame.pack(side=tk.RIGHT, anchor="n", padx=(15, 0))

    # Declare global variables
    global height_entry, width_entry, num_channels_entry, use_saved_dims_var
    global em_entry, conductance_entry  # NEW GLOBALS

    use_saved_dims_var = tk.BooleanVar(value=False)  # Default unchecked

    def toggle_entry_state():
        """
        Handle checkbox toggle for using saved dimensions.
        - If checked: auto-fill fields and disable them.
        - If unchecked: clear fields and enable them.
        """
        if use_saved_dims_var.get():
            try:
                height = nanochannel_dimensions.get("height")
                width = nanochannel_dimensions.get("width")
                number = nanochannel_dimensions.get("number")

                if None in (height, width, number):
                    raise ValueError("Missing dimension(s) in nanochannel_dimensions.")

                height_entry.config(state=tk.NORMAL)
                width_entry.config(state=tk.NORMAL)
                num_channels_entry.config(state=tk.NORMAL)

                height_entry.delete(0, tk.END)
                height_entry.insert(0, str(height))

                width_entry.delete(0, tk.END)
                width_entry.insert(0, str(width))

                num_channels_entry.delete(0, tk.END)
                num_channels_entry.insert(0, str(int(number)))

                height_entry.config(state=tk.DISABLED)
                width_entry.config(state=tk.DISABLED)
                num_channels_entry.config(state=tk.DISABLED)

            except Exception as e:
                messagebox.showerror("Error", f"Could not load saved dimensions:\n{e}")
        else:
            height_entry.config(state=tk.NORMAL)
            width_entry.config(state=tk.NORMAL)
            num_channels_entry.config(state=tk.NORMAL)

            height_entry.delete(0, tk.END)
            width_entry.delete(0, tk.END)
            num_channels_entry.delete(0, tk.END)

    # Section title
    tk.Label(power_frame, text="Power Density Calc", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

    # Checkbox to toggle saved/manual input
    tk.Checkbutton(
        power_frame,
        text="Use saved device dimensions",
        variable=use_saved_dims_var,
        command=toggle_entry_state
    ).pack(anchor="w", pady=(0, 4))

    # Geometry input: n (channels)
    row1 = tk.Frame(power_frame)
    row1.pack(anchor="w", pady=1)
    tk.Label(row1, text="n (channels):", width=14).pack(side=tk.LEFT)
    num_channels_entry = tk.Entry(row1, width=12)
    num_channels_entry.pack(side=tk.LEFT)

    # Geometry input: h (m)
    row2 = tk.Frame(power_frame)
    row2.pack(anchor="w", pady=1)
    tk.Label(row2, text="h (m):", width=14).pack(side=tk.LEFT)
    height_entry = tk.Entry(row2, width=12)
    height_entry.pack(side=tk.LEFT)

    # Geometry input: w (m)
    row3 = tk.Frame(power_frame)
    row3.pack(anchor="w", pady=1)
    tk.Label(row3, text="w (m):", width=14).pack(side=tk.LEFT)
    width_entry = tk.Entry(row3, width=12)
    width_entry.pack(side=tk.LEFT)

    # Manual input: Em (V)
    row4 = tk.Frame(power_frame)
    row4.pack(anchor="w", pady=1)
    tk.Label(row4, text="Em (V):", width=14).pack(side=tk.LEFT)
    em_entry = tk.Entry(row4, width=12)
    em_entry.pack(side=tk.LEFT)

    # Manual input: Conductance (S)
    row5 = tk.Frame(power_frame)
    row5.pack(anchor="w", pady=1)
    tk.Label(row5, text="Conductance (S):", width=14).pack(side=tk.LEFT)
    conductance_entry = tk.Entry(row5, width=12)
    conductance_entry.pack(side=tk.LEFT)

    # Button to trigger power density calculation
    tk.Button(
        power_frame,
        text="Calculate Max Power Density",
        command=lambda: calculate_power_density_from_table(),
        width=24
    ).pack(pady=(6, 0))

    # Initial call
    toggle_entry_state()
 
    tk.Label(conversion_frame, text="Convert Value").grid(row=0, column=0, columnspan=2)
    tk.Label(conversion_frame, text="Value:").grid(row=1, column=0)
    value_entry = tk.Entry(conversion_frame, width=14)
    value_entry.grid(row=1, column=1)
    tk.Label(conversion_frame, text="Concentration (mol/L):").grid(row=2, column=0)
    conc_entry = tk.Entry(conversion_frame, width=14)
    conc_entry.grid(row=2, column=1)
    tk.Label(conversion_frame, text="From Unit:").grid(row=3, column=0)
    unit_options = ["S/m", "mS/cm", "μS/cm", "mS/m", "μS/m", "S·cm²/mol", "S·m²/mol"]

    from_unit = ttk.Combobox(conversion_frame, values=unit_options, width=12)
    from_unit.grid(row=3, column=1)
    tk.Label(conversion_frame, text="To Unit:").grid(row=4, column=0)
    to_unit = ttk.Combobox(conversion_frame, values=unit_options, width=12)
    to_unit.grid(row=4, column=1)
    tk.Button(conversion_frame, text="Transfer",
          command=lambda: convert_units(value_entry, conc_entry, from_unit, to_unit, result_label),
          width=14).grid(row=5, column=0)

    result_label = tk.Entry(conversion_frame, width=14)
    result_label.grid(row=5, column=1)

    # 表格
    cols = ("Concentration", "Total Potential", "Redox Potential", "Em", "μ⁺/μ⁻", "μ⁺", "μ⁻")
    tree = ttk.Treeview(drift_diffusion_table, columns=cols, show="headings", height=18)
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, anchor="center")
    tree.pack(fill=tk.X, padx=10, pady=(0, 10))
    def make_treeview_editable(tree, columns):
        def on_double_click(event):
            item = tree.identify_row(event.y)
            column = tree.identify_column(event.x)
            if not item or not column:
                return

            col_index = int(column[1:]) - 1
            if col_index < 0:
                return

            x, y, width, height = tree.bbox(item, column)
            value = tree.set(item, columns[col_index])

            entry = tk.Entry(tree)
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, value)
            entry.focus()

            def save_edit(event):
                tree.set(item, columns[col_index], entry.get())
                entry.destroy()

            entry.bind("<Return>", save_edit)
            entry.bind("<FocusOut>", lambda e: entry.destroy())

        tree.bind("<Double-1>", on_double_click)

    make_treeview_editable(tree, cols)

    # 统一底部按钮布局：Update 单独一行，居中在上方
    bottom_wrapper = tk.Frame(drift_diffusion_table)
    bottom_wrapper.pack(side=tk.BOTTOM, pady=(10, 10))

    # 两个按钮同一行，并排居中
    update_compute_row = tk.Frame(bottom_wrapper)
    update_compute_row.grid(row=0, column=0, columnspan=4, pady=(0, 8))

    tk.Button(update_compute_row, text="Compute Monovalent Mobility", command=update_single_row).pack(side=tk.LEFT, padx=10)
    tk.Button(update_compute_row, text="Compute General Salt Mobility", command=compute_nonmonovalent).pack(side=tk.LEFT, padx=10)
    



    # 其余按钮一行居中排列
    tk.Button(bottom_wrapper, text="Save Table", command=save_table_to_csv).grid(row=1, column=0, padx=6)
    tk.Button(bottom_wrapper, text="Clear Table", command=clear_table).grid(row=1, column=1, padx=6)
    tk.Button(bottom_wrapper, text="Delete Row", command=delete_selected_row).grid(row=1, column=2, padx=6)
    formula_frame = tk.Frame(bottom_wrapper)
    formula_frame.grid(row=2, column=0, columnspan=5, pady=6)
    tk.Button(formula_frame, text="Show Redox Potential Formula", command=lambda: show_formula_popup(
    r'$E_{\mathrm{redox}} = \dfrac{k_B T}{e} \ln \left( \dfrac{\gamma_H C_H}{\gamma_L C_L} \right)$',
    title="Redox Potential Formula"
)).pack(side=tk.LEFT, padx=5)

    tk.Button(formula_frame, text="Show Monovalent Formula", command=lambda: show_formula_popup(
    r'$E_m = E_{\mathrm{tot}} - E_{\mathrm{redox}}$' + '\n' +
    r'$\Delta C = \dfrac{C_H}{C_L}$' + '\n' +
    r'$\dfrac{\mu^+}{\mu^-} = \dfrac{\ln \left( \dfrac{C_H}{C_L} \right) + \dfrac{e E_m}{k_B T}}{\ln \left( \dfrac{C_H}{C_L} \right) - \dfrac{e E_m}{k_B T}}$' + '\n' +
    r'$\mu^+ + \mu^- = \dfrac{\sigma}{e N_A C_H \cdot 1000}$' + '\n' +
    r'$\mu^- = \dfrac{\mu^+ + \mu^-}{1 + \mu^+ / \mu^-}$',
    title="Monovalent Drift-Diffusion Formula"
)).pack(side=tk.LEFT, padx=5)
    tk.Button(formula_frame, text="Show General Salt Formula", command=lambda: show_formula_popup(
    r'$E_{\mathrm{m}} = E_{\mathrm{tot}} - E_{\mathrm{redox}}$' + '\n' +
    r'$\ln \Delta = \ln \left( \dfrac{C_H}{C_L} \right)$' + '\n' +
    r'$\dfrac{\mu^+}{\mu^-} = \mathrm{abs} \left( \dfrac{z^+}{z^-} \cdot \dfrac{ \ln \Delta  - \dfrac{z^- F E_{\mathrm{m}}}{R T} }{ \ln \Delta  - \dfrac{z^+ F E_{\mathrm{m}}}{R T} } \right)$' + '\n' +
    r'$\sigma = F \left( |z^+| \mu^+ C^+ + |z^-| \mu^- C^- \right)$' + '\n' +
    r'$\mu^- = \dfrac{\sigma}{F \left( |z^+| \cdot \mu^+/\mu^- \cdot C^+ + |z^-| C^- \right)} \div 1000$',
    title="General Salt Drift-Diffusion Formula"
)).pack(side=tk.LEFT, padx=5)





   


"""
def show_electrolyte_table():
    for widget in right_frame.winfo_children():
        widget.destroy()

    frame = tk.Frame(right_frame)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    tk.Label(frame, text="Electrolyte Conductivity Table", font=("Arial", 14, "bold")).pack(pady=5)

    # 初始数据
    data = [
        ["KCl", 1.0, 9.83],
        ["KCl", 0.1, 1.289],
        ["KCl", 0.01, 0.14348],
        ["KCl", 1e-3, 0.014688],
        ["KCl", 1e-4, 0.0014979],
        ["KCl", 1e-5, 0.00014979],
        ["KCl", 1e-6, 0.000014979],
        ["LiCl", 1.0, 6.34],
        ["LiCl", 0.1, 0.9581],
        ["LiCl", 0.01, 0.10727],
        ["LiCl", 1e-3, 0.011234],
        ["LiCl", 1e-4, 0.00115],
        ["LiCl", 1e-5, 0.000115],
        ["LiCl", 1e-6, 1.15e-5],
    ]

    columns = ("Salt", "Concentration_M", "Conductivity_S_per_m")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=180)
    tree.pack(fill=tk.BOTH, expand=True)

    for row in data:
        tree.insert("", "end", values=row)

    # 支持主表双击编辑
    def on_double_click(event):
        item = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if not item or not column:
            return
        col_idx = int(column[1:]) - 1
        if col_idx < 0:
            return
        x, y, width, height = tree.bbox(item, column)
        entry = tk.Entry(tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, tree.set(item, column))

        def save_edit(e):
            tree.set(item, column, entry.get())
            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", lambda e: entry.destroy())

    tree.bind("<Double-1>", on_double_click)

    



    # 主按钮区添加这个按钮
    tk.Button(frame, text="Select or input the one you use", command=lambda: open_user_selection_window(tree)).pack(pady=10)
    def save_table_to_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
        if not file_path:
            return
        df_out = pd.DataFrame([tree.item(row)['values'] for row in tree.get_children()],
                          columns=["Salt", "Concentration_M", "Conductivity_S_per_m"])
        df_out.to_csv(file_path, index=False)
        messagebox.showinfo("Saved", f"Table saved to:\n{file_path}")

    def clear_table():
        for row in tree.get_children():
            tree.delete(row)

    def delete_selected_row():
        selected = tree.selection()
        if not selected:
            return
        for item in selected:
            tree.delete(item)
    button_frame = tk.Frame(frame)
    button_frame.pack(pady=10)

    tk.Button(button_frame, text="Save Table", command=save_table_to_csv).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Clear Table", command=clear_table).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Delete Selected Row", command=delete_selected_row).pack(side=tk.LEFT, padx=5)
"""

# %% [10] Formula Display Utilities
# Formula
def show_formula_popup(latex_formula: str, title: str = "Formula"):
    import tkinter as tk
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    popup = tk.Toplevel()
    popup.title(title)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.text(0.5, 0.5, latex_formula, fontsize=18, ha='center', va='center')
    ax.axis('off')

    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

    tk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)

def render_latex_formula(latex, dpi=100):
    fig = plt.figure(figsize=(4, 1), dpi=dpi)  # Reduced size and DPI
    plt.text(0.5, 0.5, latex, fontsize=16, ha='center', va='center')
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer)

# %% [11] Reference List Display Utilities
def show_reference_window():
    global reference_window

    if 'reference_window' in globals() and reference_window.winfo_exists():
        reference_window.lift()
        return

    reference_window = tk.Toplevel()
    reference_window.title("Reference List")
    reference_window.geometry("800x500")

    paned = tk.PanedWindow(reference_window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
    paned.pack(fill=tk.BOTH, expand=True)

    listbox = tk.Listbox(paned, width=45)
    paned.add(listbox)

    right_frame = tk.Frame(paned)
    paned.add(right_frame)

    title_label = tk.Label(right_frame, text="", font=("Helvetica", 12, "bold"), anchor="w", justify="left")
    title_label.pack(anchor="nw", pady=(10, 5), padx=10)

    content_text = tk.Text(right_frame, wrap="word", font=("Helvetica", 10))
    content_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    content_text.config(state="disabled")

    # === Reference Dictionary ===
    references = {
        "1. Electrolyte Conductivity Table": {
            "title": "1. Electrolyte Conductivity Table",
            "content": """This reference provides standard ionic conductivity (σ) values for electrolytes like KCl over a wide range of temperatures and concentrations.

The conductivity of the LiCl, KCl lower than 0.1M is from the CRC hand book.

The coductivity of 1M KCl and 1M LiCl is obtained and agreed with group member. 

Sources:""",  # End here — links will be added below
        },

        "2. Theoretical Conductance Calculation": {
    "title": "2. Theoretical Conductance Calculation",
    "content": """The theoretical conductance of a slit-like nanochannel is given by:

    G = σ × (width × height / length) × channel_number

Where:
    G = ionic conductance (S)
    σ = ionic conductivity of the electrolyte (S/m)
    width = channel width (m)
    height = channel height (m)
    length = channel length (m)
    channel_number = number of parallel nanochannels

This formula is used to predict the ideal conductance of nanofluidic channels based on geometry and electrolyte properties.

---

**Rearranged Form (to determine experimental conductivity):**

    σ = (G × length) / (width × height × channel_number)
      = (G × length) / (area × channel_number)

This rearranged form is commonly used to calculate the **experimental ionic conductivity σ**, by:

1. Measuring G from the slope of the I–V curve (unit: S)
2. Using known geometric dimensions of the nanochannel
3. Inputting the number of parallel channels

This is useful for comparing experimental σ with tabulated values from reference conductivity tables.

Reference:
Yi You, Abdulghani Ismail, Gwang-Hyeon Nam, Solleti Goutham, Ashok Keerthi, and Boya Radha,
"Angstrofluidics: Walking to the Limit", Annual Review of Materials Research, Vol. 52, 2022, pp. 189–218.
""",
    "link": "https://doi.org/10.1146/annurev-matsci-081320-032747"
},

"3. Polygon Area Calculation (Shoelace Formula)": {
    "title": "3. Polygon Area Calculation (Shoelace Formula)",
    "content": """""",
    "latex": r"$A = \dfrac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|$",
    "explanation": """\nWhere:
    A = polygon area (e.g., hysteresis area)
    (x_i, y_i) = coordinates of the ith point in the curve (ordered)

The sign of the area indicates the direction of traversal:
- Counter-clockwise → positive
- Clockwise → negative

In practice, we take the absolute value for plotting and comparison.\n
This implementation follows the method described by *OriginLab* for calculating polygonal area using the Shoelace formula.  

Reference: [OriginLab – Math Function: Polygon Area](https://www.originlab.com/doc/Origin-Help/Math-PolygonArea)
""",
    "link": "https://www.originlab.com/doc/Origin-Help/Math-PolygonArea"
},



"4. Area Normalization (Min-Max Scaling)": {
    "title": "4. Area Normalization (Min-Max Scaling)",
    "content": """""",
    "latex": r"$A_{\mathrm{norm}} = \dfrac{A - A_{\min}}{A_{\max} - A_{\min}}$",
    "explanation": """\nWhere:
A = measured area
A_min = minimum area among the dataset
A_max = maximum area among the dataset
A_norm = normalized area (unitless)

This normalization method is used to rescale the calculated hysteresis areas (e.g., from IV or GV loops) between 0 and 1. 
It enables consistent comparison across datasets with different magnitude ranges. 

Reference: 
Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann. See Chapter 3: Data Preprocessing, Section 3.5: Data Transformation and Data Discretization, Subsection 3.5.2: Data Transformation by Normalization.
""",
    "link": "https://www.sciencedirect.com/science/article/pii/B9780123814791000034#s0135"
},
    
"5. Redox Potential and Osmotic Potential": {
    "title": "5. Redox Potential and Osmotic Potential",
    "content": """""",
    "latex": r"$E_{\mathrm{redox}} = \frac{k_B T}{e} \ln\left(\frac{\gamma_H C_H}{\gamma_L C_L}\right)$",

    "explanation": """\nWhere:
    E_redox = redox potential due to concentration asymmetry  
    k_B = Boltzmann constant  
    T = temperature (K)  
    e = elementary charge  
    γ_H, γ_L = activity coefficients (high and low concentration sides)  
    C_H, C_L = ion concentrations (high and low sides)

This formula corrects the open-circuit potential (E_oc) to extract the osmotic potential (E_osm):  
    E_osm = E_oc − E_redox

Reference:  
Yi You, Abdulghani Ismail, Gwang-Hyeon Nam, Solleti Goutham, Ashok Keerthi, and Boya Radha,  
*Angstrofluidics: Walking to the Limit*, Annual Review of Materials Research, Vol. 52, 2022, pp. 189–218.
""",
    "link": "https://doi.org/10.1146/annurev-matsci-081320-032747"
},

"6. Mobility Calculation for Monovalent Salts": {
    "title": "6. Mobility Calculation for Monovalent Salts",
    "content": """""",
    "latex": r"$\dfrac{\mu^+}{\mu^-} = \dfrac{\ln(\Delta C) + \dfrac{eE_m}{k_B T}}{\ln(\Delta C) - \dfrac{eE_m}{k_B T}}$",

    "explanation": """\nWhere:
    μ⁺ / μ⁻ = ion mobility ratio (cation to anion)  
    ΔC = concentration ratio C_H / C_L  
    Eₘ = zero-current (drift-diffusion) potential  
    k_B = Boltzmann constant  
    T = temperature (K)  
    e = elementary charge  

This Henderson-based equation applies to monovalent salts in confined nanofluidic systems.

Reference:  
Solleti Goutham, Ashok Keerthi, Abdulghani Ismail, Ankit Bhardwaj, Hossein Jalali, Yi You, Yiheng Li,  
Nasim Hassani, Haoke Peng, Marcos Vinicius Surmani Martins, Fengchao Wang, Mehdi Neek-Amal,  
Boya Radha.  
*Beyond steric selectivity of ions using ångström-scale capillaries*,  
**Nature Nanotechnology**, 2023.  
""",
    "link": "https://doi.org/10.1038/s41565-023-01337-y"
},

"7. Mobility Calculation for General Salts": {
    "title": "7. Mobility Calculation for General Salts",
    "content": """

- The concentration ratio is expressed as:
  lnΔ = ln(C_H / C_L)

- The mobility ratio is derived from:
  μ⁺ / μ⁻ = abs[(z⁻ / z⁺) · (lnΔ − (z⁻·F·Eₘ)/(R·T)) / (lnΔ − (z⁺·F·Eₘ)/(R·T))]

- Conductivity relation:
  σ = F (|z⁺|·|μ⁺|·C⁺ + |z⁻|·|μ⁻|·C⁻)

- Absolute μ⁻:
  μ⁻ = [(μ⁺ / μ⁻ + 1)⁻¹] · (σ / [F (|z⁺|·μ⁺/μ⁻·C⁺ + |z⁻|·C⁻)]) · 1000

This set of equations allows computing both μ⁺ and μ⁻ from σ, Em, z⁺, z⁻, and known concentrations.

Note 1000 is calculated as 
""",
   "latex": r"$\dfrac{\mu^+}{\mu^-} = \left| \dfrac{z^-}{z^+} \cdot \dfrac{\ln \Delta - \dfrac{z^+ F E_m}{RT}}{\ln \Delta - \dfrac{z^- F E_m}{RT}} \right|$",

    "explanation": """
- μ⁺, μ⁻: cation and anion mobilities  
- z⁺, z⁻: valences of cation and anion  
- C_H, C_L: high and low concentrations  
- Eₘ: zero-current potential  
- R: gas constant  
- T: absolute temperature  
- F: Faraday constant  
- σ: conductivity  

This approach generalizes mobility computation beyond monovalent cases.

Reference:  
A. Esfandiar, B. Radha, F. C. Wang, et al.,  
*Size effect in ion transport through angstrom-scale slits*,  
**Science**, Vol. 358, Issue 6362, 2017, pp. 511–513.  
DOI: [10.1126/science.aan5275](https://doi.org/10.1126/science.aan5275)
""",
    "link": "https://doi.org/10.1126/science.aan5275"
},

"8. Maximum Power Density Calculation": {
    "title": "8. Maximum Power Density Calculation",
    "content": "",

    "latex": r"$P_{\mathrm{osm}} = \frac{1}{4} \cdot G \cdot E_m^2,\quad P_{\mathrm{density}} = \frac{P_{\mathrm{osm}}}{A_{\mathrm{eff}}} = \frac{1}{4} \cdot \frac{G \cdot E_m^2}{n \cdot h \cdot w}$",

    "explanation": """
- G: Conductance (S)  
- Em: Zero-current potential (V)  
- n: Number of nanochannels  
- h: Height of each nanochannel (m)  
- w: Width of each nanochannel (m)

The numerator (¼ · G · Em²) corresponds to the maximum osmotic power under a resistive load.  
The denominator (n · h · w) gives the effective cross-sectional area A_eff.

This equation provides a direct method to estimate the theoretical power generation potential of nanochannel membranes.

Reference:  
A. Bhardwaj, R. K. Gogoi, W. J. Howard, et al.,  
*Ultramicrotomy-Assisted Fabrication of Nanochannels for Efficient Ion Transport and Energy Generation*,  
**Advanced Functional Materials**, 2024.  
See Supplementary Info, Page 20.
""",

    "link": "https://advanced.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fadfm.202401988&file=adfm202401988-sup-0001-SuppMat.pdf"
}






}

    for key in references:
        listbox.insert(tk.END, key)

    def on_select(event):
        selection = listbox.curselection()
        if not selection:
            return

        key = listbox.get(selection[0])
        ref = references[key]

        title_label.config(text=ref["title"])

        content_text.config(state="normal")
        content_text.delete("1.0", tk.END)

        # Insert content
        content_text.insert("1.0", ref["content"] + "\n\n")

        if key == "1. Electrolyte Conductivity Table":
            # Insert each source line individually
            content_text.insert(tk.END, "- Masaryk University: “Conductivity of Electrolytes” (PDF)\n")
            content_text.insert(tk.END, "- FreeChemistry.ru: “KCl Conductivity Table” (web)\n")

            # Tag lines for hyperlink behavior
            masaryk_start = content_text.index("end-3l linestart")
            masaryk_end = content_text.index("end-3l lineend")
            content_text.tag_add("masaryk_link", masaryk_start, masaryk_end)
            content_text.tag_config("masaryk_link", foreground="blue", underline=1)
            content_text.tag_bind("masaryk_link", "<Button-1>", lambda e: webbrowser.open_new("https://is.muni.cz/el/sci/podzim2016/C4020/um/pom/Conductivity_of_Electrolytes.pdf"))
            content_text.tag_bind("masaryk_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("masaryk_link", "<Leave>", lambda e: content_text.config(cursor=""))

            freechem_start = content_text.index("end-2l linestart")
            freechem_end = content_text.index("end-2l lineend")
            content_text.tag_add("freechem_link", freechem_start, freechem_end)
            content_text.tag_config("freechem_link", foreground="blue", underline=1)
            content_text.tag_bind("freechem_link", "<Button-1>", lambda e: webbrowser.open_new("https://www.freechemistry.ru/anotes/refer/udkcl.htm"))
            content_text.tag_bind("freechem_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("freechem_link", "<Leave>", lambda e: content_text.config(cursor=""))

        elif key == "2. Theoretical Conductance Calculation":
            # Add single link at the end
            content_text.insert(tk.END, "Click here to open the reference link.")
            start = content_text.index("end-1l linestart")
            end = content_text.index("end-1l lineend")
            content_text.tag_add("link", start, end)
            content_text.tag_config("link", foreground="blue", underline=1)
            content_text.tag_bind("link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("link", "<Leave>", lambda e: content_text.config(cursor=""))
        
        elif key == "3. Polygon Area Calculation (Shoelace Formula)":
            # Insert initial content before formula
            content_text.insert("1.0", "To quantify the area enclosed by a closed curve (e.g., between up and down IV or GV sweeps), the shoelace formula is applied:\n\n")


            # === Render LaTeX inline ===
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # Prevent garbage collection

            # === Continue explanation ===
            content_text.insert(tk.END, ref["explanation"])

            # === Add clickable reference ===
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("shoelace_link", start, end)
            content_text.tag_config("shoelace_link", foreground="blue", underline=1)
            content_text.tag_bind("shoelace_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("shoelace_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("shoelace_link", "<Leave>", lambda e: content_text.config(cursor=""))



        elif key == "4. Area Normalization (Min-Max Scaling)":
            content_text.insert("1.0", 'To rescale the area A between 0 and 1 for fair comparison across devices or frequencies, min-max normalization is applied:\n\n')  # Insert description
            
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # Prevent garbage collection
            
            content_text.insert(tk.END, ref["explanation"])
            
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("norm_link", start, end)
            content_text.tag_config("norm_link", foreground="blue", underline=1)
            content_text.tag_bind("norm_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("norm_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("norm_link", "<Leave>", lambda e: content_text.config(cursor=""))


        elif key == "5. Redox Potential and Osmotic Potential":
            content_text.insert("1.0", "To calculate the redox potential due to asymmetric salt concentration (e.g., across a nanochannel), the following equation is used:\n\n")

            # Render LaTeX formula image
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # prevent garbage collection

            # Continue with explanation
            content_text.insert(tk.END, ref["explanation"])

            # Add clickable link
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("redox_link", start, end)
            content_text.tag_config("redox_link", foreground="blue", underline=1)
            content_text.tag_bind("redox_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("redox_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("redox_link", "<Leave>", lambda e: content_text.config(cursor=""))

        elif key == "6. Mobility Calculation for Monovalent Salts":
            content_text.delete("1.0", tk.END)
            
            # Insert content (empty line for spacing if needed)
            content_text.insert("1.0", "To evaluate the relative mobility of ions in drift-diffusion experiments involving monovalent electrolytes, the Henderson equation is applied. This model relates the mobility ratio to the measured zero-current potential (Em) and concentration gradient:\n\n")
            
            # Render LaTeX formula
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # Prevent garbage collection
            
            # Insert explanation
            content_text.insert(tk.END, ref["explanation"])
            
            # Add clickable reference link
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("mobility_link", start, end)
            content_text.tag_config("mobility_link", foreground="blue", underline=1)
            content_text.tag_bind("mobility_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("mobility_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("mobility_link", "<Leave>", lambda e: content_text.config(cursor=""))

        elif key == "7. Mobility Calculation for General Salts":
            # Insert initial summary sentence
            content_text.insert("1.0", "To compute mobility ratio and absolute mobilities for general salts, the following extended formulas are used:\n\n")
            
            # === Render LaTeX inline ===
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # Prevent garbage collection
            
            # === Continue explanation ===
            content_text.insert(tk.END, ref["explanation"])
            
            # === Add clickable reference ===
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("general_salt_link", start, end)
            content_text.tag_config("general_salt_link", foreground="blue", underline=1)
            content_text.tag_bind("general_salt_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("general_salt_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("general_salt_link", "<Leave>", lambda e: content_text.config(cursor=""))

        elif key == "8. Maximum Power Density Calculation":
            content_text.insert("1.0", "To compute the theoretical maximum power density from IV curves and geometry, the following formulas are used:\n\n")
            
            # === Render LaTeX inline ===
            img = render_latex_formula(ref["latex"])
            img_tk = ImageTk.PhotoImage(img)
            content_text.image_create(tk.END, image=img_tk)
            content_text.image = img_tk  # Prevent garbage collection
            
            # === Continue explanation ===
            content_text.insert(tk.END, ref["explanation"])
            
            # === Add clickable reference ===
            content_text.insert(tk.END, "\nClick here to open the reference link.\n")
            start = content_text.index("end-2l linestart")
            end = content_text.index("end-2l lineend")
            content_text.tag_add("power_density_link", start, end)
            content_text.tag_config("power_density_link", foreground="blue", underline=1)
            content_text.tag_bind("power_density_link", "<Button-1>", lambda e: webbrowser.open_new(ref["link"]))
            content_text.tag_bind("power_density_link", "<Enter>", lambda e: content_text.config(cursor="hand2"))
            content_text.tag_bind("power_density_link", "<Leave>", lambda e: content_text.config(cursor=""))


    listbox.bind("<<ListboxSelect>>", on_select)


# %% [12] Main GUI Layout & Application Entry Point
# >>> added
import sys
# optional but helpful if you open figures elsewhere in the app
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
# <<< added

startup_quotes = [
    
    "Click 'OK' to start successful research.",
    "Press OK to drink the magic coffee to start a successful research",
    "Click 'OK' to launch your path to Nature.",
    "Today is a good day to make breakthroughs!",
    "One click closer to your next publication."
]
exit_quotes = [
    "Please take a good rest, you’ve done an amazing job — one step closer to Nature!",
    "Great work today! Recharge and come back stronger.",
    "You’ve pushed the boundaries of science — now it’s time to rest.",
    "Fantastic progress! Keep going, success is within reach."
]

def show_intro_popup(parent):
    win = tk.Toplevel(parent)          # <-- parented popup; no ghost window
    win.title("Research Startup Confirmation")
    win.resizable(False, False)

    emoji = tk.Label(win, text="☕", font=("Segoe UI Emoji", 32))
    emoji.pack(pady=(12, 0))

    msg = tk.Label(win, text=random.choice(startup_quotes),
                   font=("Segoe UI", 11), wraplength=320, justify="center")
    msg.pack(padx=16, pady=8)

    btns = tk.Frame(win); btns.pack(pady=(0, 12))
    ok = tk.Button(btns, text="OK", width=10, command=win.destroy)
    ok.pack(side=tk.LEFT, padx=6)

    cancel_pressed = {"value": False}
    def cancel():
        cancel_pressed["value"] = True
        win.destroy()
    tk.Button(btns, text="Cancel", width=10, command=cancel).pack(side=tk.LEFT, padx=6)

    # make modal + center
    win.grab_set()
    win.update_idletasks()
    w, h = win.winfo_width(), win.winfo_height()
    x = (win.winfo_screenwidth() - w) // 2
    y = (win.winfo_screenheight() - h) // 3
    win.geometry(f"+{x}+{y}")
    parent.wait_window(win)
    return not cancel_pressed["value"]

# >>> added: unified quit helper
def _quit_app():
    """Break mainloop, destroy Tk resources, close figures, and exit the process."""
    try:
        # close any child windows
        try:
            for w in list(root.winfo_children()):
                try:
                    w.destroy()
                except Exception:
                    pass
        except Exception:
            pass

        # close matplotlib figures if used
        try:
            if plt is not None:
                plt.close('all')
        except Exception:
            pass

        # stop tk loop and destroy root
        try:
            root.quit()      # break mainloop
        except Exception:
            pass
        try:
            root.destroy()   # free tk resources
        except Exception:
            pass
    finally:
        # ensure the Python process (or Spyder run) ends so console isn't left running
        sys.exit(0)
# <<< added
# >>> modified: call unified quit
def on_gui_close():
    if messagebox.askokcancel("Exit Confirmation", random.choice(exit_quotes)):
        _quit_app()
# <<< modified
# ==== Start Application ====
root = tk.Tk()
root.withdraw()                         # hide until confirmed



# ==== Start Application ====
if show_intro_popup(root):  # Only run if user clicks "Confirm"
    # GUI layout (改为左右结构)
    root.deiconify()
    root.title("Nanochannel Analysis GUI")
    root.geometry("600x420")  # 宽度加大，方便左右排布

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # 左侧按钮栏
    left_frame = tk.Frame(main_frame, padx=10, pady=10, bg="lightgray")
    left_frame.pack(side=tk.LEFT, fill=tk.Y)

    # 右侧展示区域
    right_frame = tk.Frame(main_frame, padx=10, pady=10)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # 左栏按钮依次放入 left_frame
    btn_nano_details = tk.Button(left_frame, text="Enter Nanochannel Details", command=show_nanochannel_inputs, width=40)
    btn_nano_details.pack(pady=4)

    btn_electrolyte = tk.Button(left_frame, text="Enter Electrolyte Details and Analysis", command=lambda: enter_electrolyte_details(right_frame), width=40)
    btn_electrolyte.pack(pady=4)

    btn_load = tk.Button(left_frame, text="Run I-V Analysis", command=load_files_and_compute_conductance, width=40)
    btn_load.pack(pady=4)

    btn_analysis = tk.Button(left_frame, text="Analysis about Concentration", command=open_analysis_ui, width=40)
    btn_analysis.pack(pady=4)

    # Drift diffusion
    btn_drift = tk.Button(
        left_frame,
        text="Run Drift-Diffusion Experiment",
        command=lambda: run_drift_diffusion_experiment(root, right_frame),
        width=40
    )
    btn_drift.pack(pady=4)

    btn_reference = tk.Button(left_frame, text="Show Reference List", command=show_reference_window, width=40)
    btn_reference.pack(pady=4)

    text_output = None

    # Bind exit confirmation
    root.protocol("WM_DELETE_WINDOW", on_gui_close)

    root.mainloop()
    # Safety: if we ever fall through mainloop (e.g., normal quit path),
    # cleanly exit the process so console is not left running.
    sys.exit(0)
else:
    # Startup cancelled — destroy hidden root and exit process
    try:
        root.destroy()
    finally:
        sys.exit(0)


