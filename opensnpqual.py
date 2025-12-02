#!/usr/bin/env python3
"""
OpenSNPQual - S-Parameter Quality Evaluation Tool
Evaluates S-parameter quality metrics including Passivity, Reciprocity, and Causality

SPDX-License-Identifier: BSD-3-Clause
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
import webbrowser

# For S-parameter processing
import skrf as rf
from scipy import signal, fft
from scipy.interpolate import interp1d
from ieee370_implementation.ieee_p370_quality_freq_domain import quality_check_frequency_domain
from ieee370_implementation.ieee_p370_quality_time_domain import quality_check

# Version information
OPENSNPQUAL_VERSION = "v0.1"  # Change xx to your desired version number


class SParameterQualityMetrics:
    """Calculate S-parameter quality metrics based on IEEE P370 standards"""
    
    def __init__(self):
        # -------------------------------------------------------------------
        # Initial (Frequency-domain) thresholds in %, higher = better
        # PQMi/RQMi
        #   GOOD:         (99.9, 100]
        #   ACCEPTABLE:   (99, 99.9]
        #   INCONCLUSIVE: (80, 99]
        #   POOR:         [0, 80]
        #
        # CQMi
        #   GOOD:         (80, 100]
        #   ACCEPTABLE:   (50, 80]
        #   INCONCLUSIVE: (20, 50]
        #   POOR:         [0, 20]
        # -------------------------------------------------------------------
        self.freq_thresholds = {
            'passivity':    {'good': 99.9, 'acceptable': 99.0, 'inconclusive': 80.0},
            'reciprocity':  {'good': 99.9, 'acceptable': 99.0, 'inconclusive': 80.0},
            'causality':    {'good': 80.0,  'acceptable': 50.0, 'inconclusive': 20.0},
        }

        # -------------------------------------------------------------------
        # Application-based (Time-domain) thresholds in mV, lower = better
        # PQMa/RQMa/CQMa
        #   GOOD:         [0, 5)
        #   ACCEPTABLE:   [5, 10)
        #   INCONCLUSIVE: [10, 15)
        #   POOR:         [15, +∞)
        # -------------------------------------------------------------------
        self.time_thresholds = {
            'passivity':    {'good': 5.0, 'acceptable': 10.0, 'inconclusive': 15.0},
            'reciprocity':  {'good': 5.0, 'acceptable': 10.0, 'inconclusive': 15.0},
            'causality':    {'good': 5.0, 'acceptable': 10.0, 'inconclusive': 15.0},
        }
    
    def get_quality_level(self, metric_name: str, value: float, domain: str = "freq") -> str:
            """
            Backwards-compatible wrapper used by older code paths (e.g. save_markdown_results).

            domain = "freq" → use Initial (frequency-domain) thresholds (PQM/RQM/CQM in %)
            domain = "time" → use Application-based (time-domain) thresholds (mV)
            """
            if domain == "time":
                return self.get_time_quality_level(metric_name, value)
            else:
                return self.get_freq_quality_level(metric_name, value)

    def load_touchstone(self, filepath: str) -> Optional[rf.Network]:
        """Load touchstone file using scikit-rf"""
        try:
            network = rf.Network(filepath)
            return network
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
    
    def check_passivity(self, network: rf.Network) -> Dict[str, float]:
        """Check passivity in frequency and time domain"""
        results = {}
        
        # Frequency domain passivity check
        s_params = network.s
        freq = network.f
        
        # Calculate singular values for each frequency point
        passivity_violations = []
        for i in range(len(freq)):
            s_matrix = s_params[i, :, :]
            singular_values = np.linalg.svd(s_matrix, compute_uv=False)
            max_sv = np.max(singular_values)
            if max_sv > 1.0:
                passivity_violations.append(max_sv - 1.0)
        
        if passivity_violations:
            results['freq_domain'] = np.max(passivity_violations)
        else:
            results['freq_domain'] = 0.0
        
        # Time domain passivity check using energy consideration
        # Convert to time domain
        time_response = self._to_time_domain(network)
        if time_response is not None:
            energy_ratio = np.sum(np.abs(time_response)**2) / len(time_response)
            results['time_domain'] = max(0, energy_ratio - 1.0) if energy_ratio > 1.0 else 0.0
        else:
            results['time_domain'] = -1  # Error indicator
        
        return results
    
    def check_reciprocity(self, network: rf.Network) -> Dict[str, float]:
        """Check reciprocity (S_ij = S_ji) in frequency and time domain"""
        results = {}
        
        # Frequency domain reciprocity check
        s_params = network.s
        n_ports = s_params.shape[1]
        
        reciprocity_errors = []
        for i in range(n_ports):
            for j in range(i+1, n_ports):
                error = np.max(np.abs(s_params[:, i, j] - s_params[:, j, i]))
                reciprocity_errors.append(error)
        
        results['freq_domain'] = np.max(reciprocity_errors) if reciprocity_errors else 0.0
        
        # Time domain reciprocity check
        time_response = self._to_time_domain(network)
        if time_response is not None and n_ports > 1:
            # Simple check: compare impulse responses
            results['time_domain'] = results['freq_domain']  # Simplified for now
        else:
            results['time_domain'] = results['freq_domain']
        
        return results
    
    def check_causality(self, network: rf.Network) -> Dict[str, float]:
        """Check causality using Hilbert transform relationship"""
        results = {}
        
        # Frequency domain causality check using Kramers-Kronig relations
        s_params = network.s
        freq = network.f
        
        causality_errors = []
        for i in range(s_params.shape[1]):
            for j in range(s_params.shape[2]):
                s_ij = s_params[:, i, j]
                
                # Check if imaginary part satisfies Hilbert transform of real part
                real_part = np.real(s_ij)
                imag_part = np.imag(s_ij)
                
                # Simplified causality check
                if len(freq) > 10:
                    # Use derivative to check smoothness
                    d_real = np.diff(real_part)
                    d_imag = np.diff(imag_part)
                    smoothness_error = np.std(d_real) + np.std(d_imag)
                    causality_errors.append(smoothness_error)
        
        results['freq_domain'] = np.mean(causality_errors) if causality_errors else 0.0
        
        # Time domain causality check
        time_response = self._to_time_domain(network)
        if time_response is not None:
            # Check for non-causal behavior (response before t=0)
            half_len = len(time_response) // 2
            pre_response = np.sum(np.abs(time_response[:half_len-10]))
            total_response = np.sum(np.abs(time_response))
            results['time_domain'] = pre_response / total_response if total_response > 0 else 0.0
        else:
            results['time_domain'] = -1
        
        return results
    
    def _to_time_domain(self, network: rf.Network) -> Optional[np.ndarray]:
        """Convert S-parameters to time domain using IFFT"""
        try:
            # Get S11 for simplicity
            s11 = network.s[:, 0, 0]
            
            # Ensure DC and Nyquist points
            if network.f[0] != 0:
                # Extrapolate to DC
                f_extended = np.concatenate([[0], network.f])
                s11_extended = np.concatenate([[s11[0]], s11])
            else:
                f_extended = network.f
                s11_extended = s11
            
            # Perform IFFT
            time_response = fft.ifft(s11_extended)
            return np.real(time_response)
        except Exception as e:
            print(f"Error in time domain conversion: {str(e)}")
            return None
    

    def get_freq_quality_level(self, metric_name: str, value: float) -> str:
        """Return GOOD / ACCEPTABLE / INCONCLUSIVE / POOR for % metrics."""
        if value < 0:
            return "error"

        t = self.freq_thresholds[metric_name]
        # thresholds store lower bounds
        good = t['good']
        acceptable = t['acceptable']
        inconclusive = t['inconclusive']

        if value > good:
            return "good"
        elif value > acceptable:
            return "acceptable"
        elif value > inconclusive:
            return "inconclusive"
        else:
            return "poor"


    def get_time_quality_level(self, metric_name: str, value: float) -> str:
        """Return GOOD / ACCEPTABLE / INCONCLUSIVE / POOR for time (mV)."""
        if value < 0:
            return "error"

        t = self.time_thresholds[metric_name]
        # thresholds store upper bounds
        if value < t['good']:
            return "good"
        elif value < t['acceptable']:
            return "acceptable"
        elif value < t['inconclusive']:
            return "inconclusive"
        else:
            return "poor"

    
    def evaluate_file(self, filepath: str) -> Dict[str, any]:
        """
        Evaluate all quality metrics for a single file using IEEE P370
        (both frequency- and time-domain).

        Returns a dict with:
            - filename
            - passivity_freq   (PQM_i, %, Initial/Frequency table)
            - reciprocity_freq (RQM_i, %, Initial/Frequency table)
            - causality_freq   (CQM_i, %, Initial/Frequency table)
            - passivity_time   (PQM_a, mV, Application/Time table)
            - reciprocity_time (RQM_a, mV, Application/Time table)
            - causality_time   (CQM_a, mV, Application/Time table)
        """
        results = {'filename': os.path.basename(filepath)}

        # Reuse common Touchstone loader
        network = self.load_touchstone(filepath)
        if network is None:
            # Keep same error pattern as your old code
            results.update({
                'passivity_freq':    -1, 'passivity_time':    -1,
                'reciprocity_freq':  -1, 'reciprocity_time':  -1,
                'causality_freq':    -1, 'causality_time':    -1,
                'error': 'Failed to load file',
            })
            return results

        try:
            freq = network.f
            sdata = np.transpose(network.s, (1, 2, 0))  # (ports, ports, freq)
            nf = len(freq)
            port_num = network.nports

            # -----------------------------
            # Frequency-domain IEEE P370
            # -----------------------------
            # Returns % metrics: CQMi, RQMi, PQMi
            causality_freq, reciprocity_freq, passivity_freq = quality_check_frequency_domain(
                sdata, nf, port_num
            )

            # -----------------------------
            # Time-domain IEEE P370
            # -----------------------------
            # Use same defaults as GUI
            data_rate = 25.0          # Gbps
            sample_per_ui = 64
            rise_per = 0.35
            pulse_shape = 1
            extrapolation_method = 1

            causality_time_mv, reciprocity_time_mv, passivity_time_mv = quality_check(
                freq, sdata, port_num,
                data_rate,
                sample_per_ui,
                rise_per,
                pulse_shape,
                extrapolation_method,
            )

            results.update({
                'passivity_freq':     passivity_freq,
                'passivity_time':     passivity_time_mv / 2.0,
                'reciprocity_freq':   reciprocity_freq,
                'reciprocity_time':   reciprocity_time_mv / 2.0,
                'causality_freq':     causality_freq,
                'causality_time':     causality_time_mv / 2.0,
            })

        except Exception as e:
            results.update({
                'passivity_freq':    -1, 'passivity_time':    -1,
                'reciprocity_freq':  -1, 'reciprocity_time':  -1,
                'causality_freq':    -1, 'causality_time':    -1,
                'error': str(e),
            })

        return results

    def evaluate_file_frequency_only(self, filepath: str) -> Dict[str, any]:
        """
        Evaluate file with frequency domain only using IEEE P370.

        Returns:
            {
                'filename': <name>,
                'passivity_freq':   PQMi (%),
                'passivity_time':   '-',
                'reciprocity_freq': RQMi (%),
                'reciprocity_time': '-',
                'causality_freq':   CQMi (%),
                'causality_time':   '-',
                'error':            <str> (optional)
            }
        """
        results = {'filename': os.path.basename(filepath)}

        # Reuse common Touchstone loader
        network = self.load_touchstone(filepath)
        if network is None:
            results.update({
                'passivity_freq':    -1, 'passivity_time':    '-',
                'reciprocity_freq':  -1, 'reciprocity_time':  '-',
                'causality_freq':    -1, 'causality_time':    '-',
                'error': 'Failed to load file',
            })
            return results

        try:
            freq = network.f
            sdata = np.transpose(network.s, (1, 2, 0))  # (ports, ports, freq)
            nf = len(freq)
            port_num = network.nports

            # Frequency domain metrics only (IEEE P370)
            causality_freq, reciprocity_freq, passivity_freq = quality_check_frequency_domain(
                sdata, nf, port_num
            )

            results.update({
                'passivity_freq':     passivity_freq,
                'passivity_time':     '-',
                'reciprocity_freq':   reciprocity_freq,
                'reciprocity_time':   '-',
                'causality_freq':     causality_freq,
                'causality_time':     '-',
            })

        except Exception as e:
            results.update({
                'passivity_freq':    -1, 'passivity_time':    '-',
                'reciprocity_freq':  -1, 'reciprocity_time':  '-',
                'causality_freq':    -1, 'causality_time':    '-',
                'error': str(e),
            })

        return results


class OpenSNPQualCLI:
    """Command-line interface for OpenSNPQual"""
    
    def __init__(self):
        self.metrics = SParameterQualityMetrics()
    
    # NEW convenience wrapper: full IEEE370 (freq + time)
    def evaluate_file_with_time_domain(self, filepath: str) -> Dict[str, any]:
        return self.metrics.evaluate_file(filepath)

    # NEW convenience wrapper: freq-only IEEE370
    def evaluate_file_frequency_only(self, filepath: str) -> Dict[str, any]:
        return self.metrics.evaluate_file_frequency_only(filepath)

    def process_csv(self, input_csv: str, output_prefix: str = None) -> str:
        """Process CSV file containing S-parameter filenames"""
        if output_prefix is None:
            output_prefix = Path(input_csv).stem
        
        # Read input CSV
        with open(input_csv, 'r') as f:
            reader = csv.reader(f)
            filenames = [row[0] for row in reader if row]
        
        # Process each file
        results = []
        for filename in filenames:
            filepath = filename.strip()
            if os.path.exists(filepath):
                result = self.metrics.evaluate_file(filepath)
                results.append(result)
            else:
                print(f"Warning: File not found - {filepath}")
        
        # Save results
        output_csv = f"{output_prefix}_result.csv"
        self.save_csv_results(results, output_csv)
        
        # Generate markdown report
        output_md = f"{output_prefix}_result.md"
        self.save_markdown_results(results, output_md)
        
        return output_csv
    
    def save_csv_results(self, results: List[Dict], output_file: str):
        """Save results to CSV file"""
        if not results:
            return
        
        fieldnames = [
            'filename',
            'passivity_freq', 'reciprocity_freq', 'causality_freq',
            'separator',
            'passivity_time', 'reciprocity_time', 'causality_time',
        ]

        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: result.get(k, '') for k in fieldnames}
                # Visual separator between FREQ and TIME columns in CSV
                row['separator'] = ''
                writer.writerow(row)

    
    def save_markdown_results(self, results: List[Dict], output_file: str):
        """Save results to Markdown file with color coding"""
        with open(output_file, 'w') as f:
            f.write(f"# OpenSNPQual {OPENSNPQUAL_VERSION}:  A Simple Quality Checker -- REPORT\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")            
                  
            # Results table
            f.write("## Results\n\n")
            f.write("| Filename | Passivity (PQMi, Freq) | Reciprocity (RQMi, Freq) | Causality (CQMi, Freq) |  | "
                    "Passivity (PQMa, Time) | Reciprocity (RQMa, Time) | Causality (CQMa, Time) |\n")
            f.write("|----------|------------------|--------------------|------------------|----|"
                    "------------------|--------------------|------------------|\n")
            
            for result in results:
                # First column is always the filename
                row = [result['filename']]

                freq_cells = []
                time_cells = []

                # Add each metric with color coding, grouped by domain
                for metric in ['passivity', 'reciprocity', 'causality']:
                    for domain, target_list in [('freq', freq_cells), ('time', time_cells)]:
                        key = f"{metric}_{domain}"
                        value = result.get(key, -1)

                        if value == '-' or value < 0:
                            target_list.append("❌ n/a")
                        else:
                            if domain == 'freq':
                                # Initial (Frequency Domain) classifier in %
                                level = self.metrics.get_freq_quality_level(metric, value)
                            else:
                                # Application-based (Time Domain) classifier in mV
                                level = self.metrics.get_time_quality_level(metric, value)

                            emoji = {
                                'good':         '🟢',
                                'acceptable':   '🔵',
                                'inconclusive': '🟡',
                                'poor':         '🔴',
                                'error':        '❌',
                            }.get(level, '❌')
                            target_list.append(f"{emoji} {value:.1f}")

                # Insert a blank separator column between FREQ and TIME sections
                row.extend(freq_cells + [""] + time_cells)

                f.write(f"| {' | '.join(row)} |\n")
            
            # Quality level legend
            
            f.write("## 📊 Quality Metrics Table - Initial (Frequency Domain) - good for quick check\n")
            f.write("\n")
            f.write("| Level | Symbol | Passivity (PQMi) | Reciprocity (RQMi)  | Causality (CQMi) | Description |\n")
            f.write("|-------|--------|-----------|-----------|-----------|-------------|\n")
            f.write("| 🟢 Good | ✓ | (99.9, 100] | (99.9, 100]  | (80, 100] | Excellent quality, suitable for critical applications |\n")
            f.write("| 🔵 Acceptable | ○ | (99, 99.9] | (99, 99.9] | (50, 80] | OK quality, may not be suitable for sensitive applications like de-embedding |\n")
            f.write("| 🟡 Inconclusive | △ | (80, 99] | (80, 99] | (20, 50] | Marginal quality, unlikely to be reliable |\n")
            f.write("| 🔴 POOR | ✗ | [0, 80] | [0, 80] | [0, 20] | Poor quality, do not use! Re-measurement (+ VNA recalibration) recommended |\n")
            f.write("\n")
            f.write("## 📊 Quality Metrics Table - Application-based (Time Domain) - rigorously computed\n")
            f.write("\n")
            f.write("| Level | Symbol | Passivity (PQMa) | Reciprocity (RQMa)  | Causality (CQMa) | Description |\n")
            f.write("|-------|--------|-----------|-----------|-----------|-------------|\n")
            f.write("| 🟢 Good | ✓ | [0 mV, 5 mV) | [0 mV, 5 mV) | [0 mV, 5 mV) | Excellent quality, suitable for critical applications |\n")
            f.write("| 🔵 Acceptable | ○ | [5 mV, 10 mV) | [5 mV, 10 mV) | [5 mV, 10 mV) | OK quality, may not be suitable for sensitive applications like de-embedding |\n")
            f.write("| 🟡 Inconclusive | △ | [10 mV, 15 mV) | [10 mV, 15 mV) | [10 mV, 15 mV) | Marginal quality, unlikely to be reliable |\n")
            f.write("| 🔴 POOR | ✗ | [15 mV, +∞) | [15 mV, +∞) | [15 mV, +∞) | Poor quality, do not use! Re-measurement (+ VNA recalibration) recommended |\n")
            f.write("\n")
            f.write("Reference:\"[IEEE Standard for Electrical Characterization of Printed Circuit Board and Related Interconnects at Frequencies up to 50 GHz,](https://ieeexplore.ieee.org/document/9316329/)\" in IEEE Std 370-2020 , vol., no., pp.1-147, 8 Jan. 2021, doi: 10.1109/IEEESTD.2021.9316329. \n")
            f.write("\n")


class CustomInfoDialog:
    """Custom dialog with clickable links and styled text"""
    
    def __init__(self, parent, title, content_func):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("500x450")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create content frame
        frame = ttk.Frame(self.dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Let the content function build the dialog content
        content_func(frame)
        
        # Add OK button
        ok_button = ttk.Button(self.dialog, text="OK", command=self.dialog.destroy)
        ok_button.pack(side=tk.BOTTOM, pady=10)
        
        # Center dialog on parent
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

class OpenSNPQualGUI:
    """GUI for OpenSNPQual"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title(f"OpenSNPQual {OPENSNPQUAL_VERSION}:  A Simple Quality Checker")
        self.root.geometry("1200x600")
        
        self.cli = OpenSNPQualCLI()
        self.file_list = []
        self.results = {}

        self.calculate_time_domain_var = None  # Will be set in setup_ui
        
        self.setup_ui()
        
        # Check if file was passed as argument
        if len(sys.argv) > 1:
            self.load_files_from_args(sys.argv[1:])
    
    def setup_ui(self):
        """Setup the GUI interface"""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load S-Parameter Files...", command=self.load_files)
        file_menu.add_command(label="Load Folder...", command=self.load_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Table", command=self.copy_table_to_clipboard)
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        
        # Add this after the Edit menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Correlation to IEEE370", command=self.show_ieee370_correlation)
        help_menu.add_command(label="Report a BUG", command=self.report_bug)
        help_menu.add_separator()
        help_menu.add_command(label="About OpenSNPQual", command=self.show_about)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Button(button_frame, text="Load SNPs", command=self.load_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate_metrics).pack(side=tk.LEFT, padx=5)
        
        # Add the checkbox for time domain calculation
        self.calculate_time_domain_var = tk.BooleanVar(value=False)  # Default unchecked
        self.time_domain_checkbox = ttk.Checkbutton(
            button_frame, 
            text="Include TIME DOMAIN",
            variable=self.calculate_time_domain_var,
            command=self.on_time_domain_toggle
        )
        self.time_domain_checkbox.pack(side=tk.LEFT, padx=15)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Table frame with scrollbars
        table_frame = ttk.Frame(main_frame)
        table_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Create Treeview for table
        # Columns are organized as: FREQ metrics | separator | TIME metrics
        columns = (
            'passivity_freq',
            'reciprocity_freq',
            'causality_freq',
            'separator',          # purely visual spacer
            'passivity_time',
            'reciprocity_time',
            'causality_time',
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show='tree headings',
            height=15,
        )

        # Define column headings
        self.tree.heading('#0', text='SNP File')
        self.tree.heading('passivity_freq', text='Passivity \n(PQMi, Freq)')
        self.tree.heading('reciprocity_freq', text='Reciprocity \n(RQMi, Freq)')
        self.tree.heading('causality_freq', text='Causality \n(CQMi, Freq)')
        # Separator column has no label – it's just a visual gap
        self.tree.heading('separator', text='')
        self.tree.heading('passivity_time', text='Passivity \n(PQMa, Time)')
        self.tree.heading('reciprocity_time', text='Reciprocity \n(RQMa, Time)')
        self.tree.heading('causality_time', text='Causality \n(CQMa, Time)')

        # Configure column widths
        self.tree.column('#0', width=300)
        for col in columns:
            if col == 'separator':
                # Narrow, non-stretching spacer between FREQ and TIME sections
                self.tree.column(col, width=20, anchor=tk.CENTER, stretch=False)
            else:
                self.tree.column(col, width=140, anchor=tk.CENTER)
        
        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout for table and scrollbars
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        
        # Define tags for color coding
        self.tree.tag_configure('great', foreground='green')
        self.tree.tag_configure('good', foreground='blue')
        self.tree.tag_configure('acceptable', foreground='orange')
        self.tree.tag_configure('bad', foreground='red')
        self.tree.tag_configure('error', foreground='gray')
    
    def on_time_domain_toggle(self):
        """Handle time domain calculation toggle"""
        if self.calculate_time_domain_var.get():
            self.status_label.config(text="Time domain calculation enabled")
        else:
            self.status_label.config(text="Time domain calculation disabled")

    def load_files(self):
        """Load S-parameter files using file dialog"""
        files = filedialog.askopenfilenames(
            title="Select S-Parameter Files",
            filetypes=[("S-Parameter files", "*.s*p"), ("All files", "*.*")]
        )
        
        if files:
            self.add_files_to_list(files)
    
    def load_folder(self):
        """Load all S-parameter files from a folder"""
        folder = filedialog.askdirectory(title="Select Folder Containing S-Parameter Files")
        
        if folder:
            # Find all .s*p files in the folder
            snp_files = []
            for ext in ['s1p', 's2p', 's3p', 's4p', 's6p', 's8p', 'snp']:
                snp_files.extend(Path(folder).glob(f"*.{ext}"))
                snp_files.extend(Path(folder).glob(f"*.{ext.upper()}"))
            
            if snp_files:
                self.add_files_to_list([str(f) for f in snp_files])
            else:
                messagebox.showwarning("No Files Found", 
                                     "No S-parameter files found in the selected folder.")
    
    def load_files_from_args(self, files):
        """Load files passed as command-line arguments"""
        self.add_files_to_list(files)
        # Auto-calculate if files were loaded from args
        self.root.after(100, self.calculate_metrics)
    
    def add_files_to_list(self, files):
        """Add files to the table"""
        for filepath in files:
            if filepath not in self.file_list:
                self.file_list.append(filepath)
                filename = os.path.basename(filepath)
                
                # Add to tree with placeholder values
                # Order: 3×FREQ, separator, 3×TIME
                item = self.tree.insert(
                    '',
                    'end',
                    text=filename,
                    values=('-', '-', '-', '', '-', '-', '-'),
                )

                
        self.status_label.config(text=f"Loaded {len(self.file_list)} files")
    
    def calculate_metrics(self):
        """Calculate metrics for all loaded files"""
        if not self.file_list:
            messagebox.showwarning("No Files", "Please load S-parameter files first.")
            return
        
        # Run calculation in separate thread to keep GUI responsive
        thread = threading.Thread(target=self._calculate_worker)
        thread.start()
    
    def _calculate_worker(self):
        """Worker thread for calculations"""
        self.status_label.config(text="Calculating...")
        self.progress_var.set(0)
        
        # Check if time domain calculation is enabled
        calculate_time_domain = self.calculate_time_domain_var.get()
        
        # Create temporary CSV file
        temp_csv = "temp_input.csv"
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for filepath in self.file_list:
                writer.writerow([filepath])

        # Process files
        total_files = len(self.file_list)
        for i, filepath in enumerate(self.file_list):
            if calculate_time_domain:
                # Use full IEEE P370 (freq + time) via CLI/backend
                result = self.cli.evaluate_file_with_time_domain(filepath)
            else:
                # Use frequency domain only via CLI/backend
                result = self.cli.evaluate_file_frequency_only(filepath)

            self.results[filepath] = result

            progress = (i + 1) / total_files * 100
            self.progress_var.set(progress)

            self.root.after(0, self._update_table_row, filepath, result)
        
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        self.status_label.config(text=f"Calculation complete for {total_files} files")
        
        # Save results automatically
        output_csv = f"snpqual_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.cli.save_csv_results(list(self.results.values()), output_csv)
        
        output_md = output_csv.replace('.csv', '.md')
        self.cli.save_markdown_results(list(self.results.values()), output_md)
        
        self.status_label.config(text=f"Results saved to {output_csv} and {output_md}")
    
    def _update_table_row(self, filepath, result):
        """Update a single row in the table"""
        filename = os.path.basename(filepath)
        
        # Find the item in the tree
        for item in self.tree.get_children():
            if self.tree.item(item)['text'] == filename:
                # Format values with quality indicators
                # Columns: FREQ metrics (passivity/reciprocity/causality),
                # then a separator column, then TIME metrics.
                freq_values = []
                time_values = []

                for metric in ['passivity', 'reciprocity', 'causality']:
                    for domain, target_list in [('freq', freq_values), ('time', time_values)]:
                        key = f"{metric}_{domain}"
                        value = result.get(key, -1)

                        if value == '-' or value < 0:
                            target_list.append("n/a")
                        else:
                            level = self.cli.metrics.get_quality_level(metric, value)
                            # Use Unicode symbols for quality levels
                            symbol = {
                                'good':         '✓',
                                'acceptable':   '○',
                                'inconclusive': '△',
                                'poor':         '✗',
                                'error':        '?',
                            }.get(level, '?')
                            target_list.append(f"{symbol} {value:.1f}")

                # Insert a blank separator between FREQ and TIME sections
                values = freq_values + [''] + time_values
                self.tree.item(item, values=values)
                
                # Determine overall quality for row coloring based on
                # the worst of ALL metrics that were actually calculated
                levels = []

                for metric in ['passivity', 'reciprocity', 'causality']:
                    # Frequency-domain metric (Initial table, %)
                    v_freq = result.get(f"{metric}_freq", -1)
                    if isinstance(v_freq, (int, float)) and v_freq >= 0:
                        levels.append(self.cli.metrics.get_freq_quality_level(metric, v_freq))

                    # Time-domain metric (Application table, mV)
                    v_time = result.get(f"{metric}_time", -1)
                    if isinstance(v_time, (int, float)) and v_time >= 0:
                        levels.append(self.cli.metrics.get_time_quality_level(metric, v_time))

                if not levels:
                    tag = 'error'
                else:
                    # Map quality labels to an ordinal, then choose the worst
                    rank = {
                        'poor': 0,
                        'inconclusive': 1,
                        'acceptable': 2,
                        'good': 3,
                        'error': -1,  # optional
                    }
                    worst_rank = min(rank.get(lvl, 0) for lvl in levels)
                    inv_rank = {v: k for k, v in rank.items()}
                    tag = inv_rank.get(worst_rank, 'poor')

                self.tree.item(item, tags=(tag,))
                break
    
    def copy_table_to_clipboard(self):
        """Copy table contents to clipboard"""
        # Build tab-separated text
        clipboard_text = "SNP File\tPassivity (Freq)\tPassivity (Time)\t" \
                        "Reciprocity (Freq)\tReciprocity (Time)\t" \
                        "Causality (Freq)\tCausality (Time)\n"
        
        for item in self.tree.get_children():
            row_data = [self.tree.item(item)['text']]
            row_data.extend(self.tree.item(item)['values'])
            clipboard_text += '\t'.join(str(v) for v in row_data) + '\n'
        
        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_text)
        self.status_label.config(text="Table copied to clipboard")
    
    def export_results(self):
        """Export results to file"""
        if not self.results:
            messagebox.showwarning("No Results", "Please calculate metrics first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Markdown files", "*.md")]
        )
        
        if filename:
            if filename.endswith('.md'):
                self.cli.save_markdown_results(list(self.results.values()), filename)
            else:
                self.cli.save_csv_results(list(self.results.values()), filename)
            
            self.status_label.config(text=f"Results exported to {filename}")
    
    def clear_all(self):
        """Clear all files and results"""
        self.file_list = []
        self.results = {}
        self.tree.delete(*self.tree.get_children())
        self.progress_var.set(0)
        self.status_label.config(text="Ready")
    
    def show_ieee370_correlation(self):
        """Show IEEE P370 correlation information"""
        def create_content(parent):
            # Title
            title_label = tk.Label(parent, text="Correlation to IEEE 370", 
                                font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 10))
            
            # Main text
            text = tk.Text(parent, wrap=tk.WORD, height=10, width=50, 
                        font=("Arial", 10))
            text.pack(fill=tk.BOTH, expand=True)
            
            # Insert content with formatting
            text.insert("1.0", "OpenSNPQual implements IEEE 370 quality metrics in Python:\n\n")
            text.insert("end", "• ", "bullet")
            text.insert("end", "Frequency Domain:", "bold")
            text.insert("end", " fully correlated to 370 code\n")
            text.insert("end", "• ", "bullet")
            text.insert("end", "Time Domain:", "bold")
            text.insert("end", " partially correlated to IEEE 370 time-domain metrics\n\n")
            text.insert("end", "Results have been validated against original IEEE370 MATLAB code.\n\n")
            text.insert("end", "For more information, visit:\n")
            text.insert("end", "Correlation Report", "link")
            
            # Configure tags
            text.tag_config("bold", font=("Arial", 10, "bold"))
            text.tag_config("bullet", foreground="#666666")
            text.tag_config("link", foreground="blue", underline=True)
            text.tag_bind("link", "<Button-1>", 
                        lambda e: webbrowser.open("https://github.com/OpenSNPTools/openSNPQual/blob/IEEEP370_Qual_Correlation/example_touchstone/sparams_info.md"))
            text.tag_bind("link", "<Enter>", lambda e: text.config(cursor="hand2"))
            text.tag_bind("link", "<Leave>", lambda e: text.config(cursor=""))
            
            text.config(state=tk.DISABLED)
        
        CustomInfoDialog(self.root, "Correlation to IEEE370", create_content)

    def report_bug(self):
        """Open bug report dialog with clickable links"""
        def create_content(parent):
            # Title
            title_label = tk.Label(parent, text="Report a Bug", 
                                font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 10))
            
            # Main text
            text = tk.Text(parent, wrap=tk.WORD, height=12, width=50, 
                        font=("Arial", 10))
            text.pack(fill=tk.BOTH, expand=True)
            
            text.insert("1.0", "To report a bug:\n\n")
            
            # Email option
            text.insert("end", "1. E-mail: ", "bold")
            text.insert("end", "giorgi.snp [at] pm [dot] me", "email_link")
            text.insert("end", "\n\n")
            
            # GitHub option
            text.insert("end", "or\n\n")
            text.insert("end", "2. GitHub Issues: ", "bold")
            text.insert("end", "OpenSNPQual Issues Page", "github_link")
            text.insert("end", "\n")
            text.insert("end", "https://github.com/OpenSNPTools/openSNPQual/issues")
            text.insert("end", "\n\n")
            
            
            text.insert("end", "Please include:\n", "bold")
            text.insert("end", "  • OpenSNPQual version\n")
            text.insert("end", "  • Description of the issue\n")
            text.insert("end", "  • Steps to reproduce\n")
            text.insert("end", "  • Error messages (if any)\n")
            text.insert("end", "  • Sample files (if applicable)")
            
            # Configure tags
            text.tag_config("bold", font=("Arial", 10, "bold"))
            text.tag_config("email_link", foreground="blue", underline=True)
            text.tag_config("github_link", foreground="blue", underline=True)
            
            # Bind click events
            text.tag_bind("email_link", "<Button-1>", 
                        lambda e: webbrowser.open("mailto:giorgi.snp@pm.me?subject=OpenSNPQual%20Bug%20Report"))
            text.tag_bind("github_link", "<Button-1>", 
                        lambda e: webbrowser.open("https://github.com/OpenSNPTools/openSNPQual/issues"))
            
            # Hover effects
            text.tag_bind("email_link", "<Enter>", lambda e: text.config(cursor="hand2"))
            text.tag_bind("email_link", "<Leave>", lambda e: text.config(cursor=""))
            text.tag_bind("github_link", "<Enter>", lambda e: text.config(cursor="hand2"))
            text.tag_bind("github_link", "<Leave>", lambda e: text.config(cursor=""))
            
            text.config(state=tk.DISABLED)
        
        CustomInfoDialog(self.root, "Report a Bug", create_content)

    def show_about(self):
        """Show about dialog with styled text"""
        def create_content(parent):
            # Logo/Title
            title_label = tk.Label(parent, text=f"OpenSNPQual {OPENSNPQUAL_VERSION}", 
                                font=("Arial", 16, "bold"), foreground="#0066cc")
            title_label.pack(pady=(0, 5))
            
            subtitle_label = tk.Label(parent, text="A Simple S-Parameter Quality Checker", 
                                    font=("Arial", 10, "italic"))
            subtitle_label.pack(pady=(0, 15))
            
            # Main text
            text = tk.Text(parent, wrap=tk.WORD, height=12, width=50, 
                        font=("Arial", 10))
            text.pack(fill=tk.BOTH, expand=True)
            
            text.insert("1.0", "A GUI tool for evaluating S-parameter quality metrics\n")
            text.insert("end", "based on IEEE 370 standard:\n")
            text.insert("end", "https://opensource.ieee.org/elec-char/ieee-370/", "website_link_370")
            text.insert("end", " \n\n")
            
            
            text.insert("end", "S-parameter Quality Metrics and features:\n", "heading")
            text.insert("end", "  • Passivity (PQM)\n")
            text.insert("end", "  • Reciprocity (RQM)\n")
            text.insert("end", "  • Causality (CQM)\n")
            text.insert("end", "  • Frequency and Time domain analysis\n")
            text.insert("end", "  • Batch file processing \n")
            text.insert("end", "  • Report generation \n\n")
            
            text.insert("end", "License: ", "bold")
            text.insert("end", "BSD 3-Clause\n")
            text.insert("end", "© 2025 Giorgi Maghlakelidze, OpenSNP Contributors, IEEE370 Contributors\n\n")
            
            text.insert("end", "Website: ", "bold")
            text.insert("end", "https://github.com/OpenSNPTools/openSNPQual/", "website_link")
            
            # Configure tags
            text.tag_config("heading", font=("Arial", 11, "bold"))
            text.tag_config("bold", font=("Arial", 10, "bold"))
            text.tag_config("website_link", foreground="blue", underline=True)
            text.tag_config("website_link_370", foreground="blue", underline=True)
            
            # Bind website link
            text.tag_bind("website_link", "<Button-1>", 
                        lambda e: webbrowser.open("https://github.com/OpenSNPTools/openSNPQual/"))
            text.tag_bind("website_link", "<Enter>", lambda e: text.config(cursor="hand2"))
            text.tag_bind("website_link", "<Leave>", lambda e: text.config(cursor=""))
            
            # Bind website link
            text.tag_bind("website_link_370", "<Button-1>", 
                        lambda e: webbrowser.open("https://opensource.ieee.org/elec-char/ieee-370/"))
            text.tag_bind("website_link_370", "<Enter>", lambda e: text.config(cursor="hand2"))
            text.tag_bind("website_link_370", "<Leave>", lambda e: text.config(cursor=""))

            text.config(state=tk.DISABLED)
        
        CustomInfoDialog(self.root, "About OpenSNPQual", create_content)

    def evaluate_file_with_time_domain(self, filepath: str) -> Dict[str, any]:
        """Evaluate file with both frequency and time domain using IEEE P370"""
        
        results = {'filename': os.path.basename(filepath)}
        
        try:
            # Load network
            network = rf.Network(filepath)
            freq = network.f
            sdata = np.transpose(network.s, (1, 2, 0))  # Convert to IEEE P370 format
            nf = len(freq)
            port_num = network.nports
            
            # Frequency domain metrics
            causality_freq, reciprocity_freq, passivity_freq = quality_check_frequency_domain(
                sdata, nf, port_num
            )
            
            # Time domain metrics (with your IEEE P370 parameters)
            data_rate = 25.0  # You might want to make this configurable
            sample_per_ui = 64
            rise_per = 0.35
            pulse_shape = 1
            extrapolation_method = 1
            
            causality_time_mv, reciprocity_time_mv, passivity_time_mv = quality_check(
                freq, sdata, port_num, data_rate, sample_per_ui, 
                rise_per, pulse_shape, extrapolation_method
            )
            
            results.update({
                # Frequency-domain metrics are 0–100 % (IEEE P370 PQMi/RQMi/CQMi)
                'passivity_freq': passivity_freq,
                'passivity_time': passivity_time_mv / 2,  # still mV, as before
                'reciprocity_freq': reciprocity_freq,
                'reciprocity_time': reciprocity_time_mv / 2,
                'causality_freq': causality_freq,
                'causality_time': causality_time_mv / 2
            })
            
        except Exception as e:
            results.update({
                'passivity_freq': -1, 'passivity_time': -1,
                'reciprocity_freq': -1, 'reciprocity_time': -1,
                'causality_freq': -1, 'causality_time': -1,
                'error': str(e)
            })
        
        return results

    def evaluate_file_frequency_only(self, filepath: str) -> Dict[str, any]:
        """
        Evaluate file with frequency domain only using IEEE P370.

        Returns:
            {
                'filename': <name>,
                'passivity_freq':   PQMi (%),
                'passivity_time':   '-',
                'reciprocity_freq': RQMi (%),
                'reciprocity_time': '-',
                'causality_freq':   CQMi (%),
                'causality_time':   '-',
                'error':            <str> (optional)
            }
        """
        results = {'filename': os.path.basename(filepath)}

        try:
            # Load network
            network = rf.Network(filepath)
            freq = network.f
            sdata = np.transpose(network.s, (1, 2, 0))  # (ports, ports, freq)
            nf = len(freq)
            port_num = network.nports

            # Frequency domain metrics only (IEEE P370)
            causality_freq, reciprocity_freq, passivity_freq = quality_check_frequency_domain(
                sdata, nf, port_num
            )

            results.update({
                
                'passivity_freq':     passivity_freq,
                'passivity_time':     '-',
                'reciprocity_freq':   reciprocity_freq,
                'reciprocity_time':   '-',
                'causality_freq':     causality_freq,
                'causality_time':     '-',
            })

        except Exception as e:
            results.update({
                'passivity_freq':    -1, 'passivity_time':    '-',
                'reciprocity_freq':  -1, 'reciprocity_time':  '-',
                'causality_freq':    -1, 'causality_time':    '-',
                'error': str(e),
            })

        return results


    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description=f"OpenSNPQual {OPENSNPQUAL_VERSION}:  A Simple Quality Checker"
    )
    parser.add_argument('--cli', action='store_true', 
                       help='Run in CLI mode')
    parser.add_argument('-i', '--input', type=str, 
                       help='Input CSV file with S-parameter filenames (CLI mode)')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output prefix for result files (CLI mode)')
    parser.add_argument('files', nargs='*', 
                       help='S-parameter files to load (GUI mode)')
    
    args = parser.parse_args()
    
    if args.cli:
        # CLI mode
        if not args.input:
            print("Error: --input is required in CLI mode")
            sys.exit(1)
        
        cli = OpenSNPQualCLI()
        output_file = cli.process_csv(args.input, args.output)
        print(f"Results saved to: {output_file}")
    else:
        # GUI mode
        app = OpenSNPQualGUI()
        app.run()


if __name__ == "__main__":
    main()
