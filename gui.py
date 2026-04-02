import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from data_loader import load_csv
from functions import remove_baseline, lowpass, highpass, bandpass, features, compute_fft

class SensorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SYSC 2010 Signal Analysis Project")
        self.root.geometry("1000x800")
        
        # State variables
        self.time = None
        self.raw_signal = None
        self.processed_signal = None
        self.fs = 100  # Default sampling rate
        
        self.setup_ui()

    def setup_ui(self):
        # Control Panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="Load CSV", command=self.open_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Sensor Type:").pack(side=tk.LEFT, padx=5)
        self.type_var = tk.StringVar()
        self.type_menu = ttk.Combobox(control_frame, textvariable=self.type_var)
        self.type_menu['values'] = ("ECG", "Temperature", "Respiration", "Motion")
        self.type_menu.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Process Signal", command=self.process).pack(side=tk.LEFT, padx=5)

        # Stats Display Panel
        self.stats_label = ttk.Label(self.root, text="Stats: Load data to see metrics", font=('Helvetica', 10, 'bold'))
        self.stats_label.pack(pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.time, self.raw_signal = load_csv(file_path)
            if self.time is not None:
                self.update_plots(raw_only=True)
                messagebox.showinfo("Success", "Data loaded successfully!")

    def process(self):
        if self.raw_signal is None:
            messagebox.showwarning("Warning", "Please load a file first.")
            return
        
        sensor_type = self.type_var.get()
        if sensor_type == "ECG":
            self.processed_signal = lowpass(self.raw_signal, self.fs, 0.1)
        elif sensor_type == "Temperature":
            self.processed_signal = lowpass(self.raw_signal, self.fs, 40)
        else:
            self.processed_signal = self.raw_signal
            
        # Update Stats
        stats = features(self.processed_signal)
        signal_range = np.max(self.process_signal) - np.min(self.processed_signal)
        self.stats_label.config(
            text=f"Mean: {stats['mean']:.2f} | "
                 f"Std: {stats['std']:.2f} | "
                 f"RMS: {stats['rms']:.2f} | "
                 f"Range: {signal_range:.2f}"
        )
        self.update_plots()

    def update_plots(self, raw_only=False):
        self.ax1.clear()
        self.ax2.clear()
        
        # Time-domain
        self.ax1.plot(self.time, self.raw_signal, label="Raw", alpha=0.5)
        if not raw_only and self.processed_signal is not None:
            self.ax1.plot(self.time, self.processed_signal, label="Processed", color='red')
        self.ax1.set_title("Time Domain")
        self.ax1.legend()
        
        # Frequency-domain (FFT)
        if not raw_only and self.processed_signal is not None:
            freqs, mag = compute_fft(self.processed_signal, self.fs)
            self.ax2.plot(freqs, mag, color='green')
            self.ax2.set_title("Frequency Domain (FFT)")
            
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorGUI(root)
    root.mainloop()