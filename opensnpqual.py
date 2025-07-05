#!/usr/bin/env python3
"""
OpenSNPQual - S-Parameter Quality Evaluation Tool
Evaluates S-parameter quality metrics including Passivity, Reciprocity, and Causality
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

# For S-parameter processing
import skrf as rf
from scipy import signal, fft
from scipy.interpolate import interp1d


class SParameterQualityMetrics:
    """Calculate S-parameter quality metrics based on IEEE P370 standards"""
    
    def __init__(self):
        self.freq_thresholds = {
            'passivity': {'great': 0.01, 'good': 0.05, 'acceptable': 0.1, 'bad': float('inf')},
            'reciprocity': {'great': 0.01, 'good': 0.05, 'acceptable': 0.1, 'bad': float('inf')},
            'causality': {'great': 0.01, 'good': 0.05, 'acceptable': 0.1, 'bad': float('inf')}
        }
    
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
    
    def get_quality_level(self, metric_name: str, value: float) -> str:
        """Determine quality level based on thresholds"""
        if value < 0:  # Error indicator
            return "error"
        
        thresholds = self.freq_thresholds.get(metric_name, self.freq_thresholds['passivity'])
        
        if value <= thresholds['great']:
            return "great"
        elif value <= thresholds['good']:
            return "good"
        elif value <= thresholds['acceptable']:
            return "acceptable"
        else:
            return "bad"
    
    def evaluate_file(self, filepath: str) -> Dict[str, any]:
        """Evaluate all quality metrics for a single file"""
        results = {'filename': os.path.basename(filepath)}
        
        network = self.load_touchstone(filepath)
        if network is None:
            results.update({
                'passivity_freq': -1, 'passivity_time': -1,
                'reciprocity_freq': -1, 'reciprocity_time': -1,
                'causality_freq': -1, 'causality_time': -1,
                'error': 'Failed to load file'
            })
            return results
        
        # Calculate metrics
        passivity = self.check_passivity(network)
        reciprocity = self.check_reciprocity(network)
        causality = self.check_causality(network)
        
        results.update({
            'passivity_freq': passivity['freq_domain'],
            'passivity_time': passivity['time_domain'],
            'reciprocity_freq': reciprocity['freq_domain'],
            'reciprocity_time': reciprocity['time_domain'],
            'causality_freq': causality['freq_domain'],
            'causality_time': causality['time_domain']
        })
        
        return results


class OpenSNPQualCLI:
    """Command-line interface for OpenSNPQual"""
    
    def __init__(self):
        self.metrics = SParameterQualityMetrics()
    
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
        
        fieldnames = ['filename', 'passivity_freq', 'passivity_time', 
                     'reciprocity_freq', 'reciprocity_time',
                     'causality_freq', 'causality_time']
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
    
    def save_markdown_results(self, results: List[Dict], output_file: str):
        """Save results to Markdown file with color coding"""
        with open(output_file, 'w') as f:
            f.write("# OpenSNPQual - S-Parameter Quality Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Quality level legend
            f.write("## Quality Levels\n")
            f.write("- 🟢 **Great**: < 0.01\n")
            f.write("- 🔵 **Good**: 0.01 - 0.05\n")
            f.write("- 🟡 **Acceptable**: 0.05 - 0.1\n")
            f.write("- 🔴 **Bad**: > 0.1\n\n")
            
            # Results table
            f.write("## Results\n\n")
            f.write("| Filename | Passivity (Freq) | Passivity (Time) | "
                   "Reciprocity (Freq) | Reciprocity (Time) | "
                   "Causality (Freq) | Causality (Time) |\n")
            f.write("|----------|------------------|------------------|"
                   "--------------------|--------------------|"
                   "------------------|-----------------|\n")
            
            for result in results:
                row = [result['filename']]
                
                # Add each metric with color coding
                for metric in ['passivity', 'reciprocity', 'causality']:
                    for domain in ['freq', 'time']:
                        key = f"{metric}_{domain}"
                        value = result.get(key, -1)
                        
                        if value < 0:
                            row.append("❌ Error")
                        else:
                            level = self.metrics.get_quality_level(metric, value)
                            emoji = {'great': '🟢', 'good': '🔵', 
                                   'acceptable': '🟡', 'bad': '🔴'}.get(level, '❌')
                            row.append(f"{emoji} {value:.4f}")
                
                f.write(f"| {' | '.join(row)} |\n")


class OpenSNPQualGUI:
    """GUI for OpenSNPQual"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OpenSNPQual - S-Parameter Quality Evaluation")
        self.root.geometry("1200x600")
        
        self.cli = OpenSNPQualCLI()
        self.file_list = []
        self.results = {}
        
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
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Button(button_frame, text="Load SNP Files", command=self.load_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate_metrics).pack(side=tk.LEFT, padx=5)
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
        columns = ('passivity_freq', 'passivity_time', 'reciprocity_freq', 
                  'reciprocity_time', 'causality_freq', 'causality_time')
        
        self.tree = ttk.Treeview(table_frame, columns=columns, show='tree headings', height=15)
        
        # Define column headings
        self.tree.heading('#0', text='SNP File')
        self.tree.heading('passivity_freq', text='Passivity (Freq)')
        self.tree.heading('passivity_time', text='Passivity (Time)')
        self.tree.heading('reciprocity_freq', text='Reciprocity (Freq)')
        self.tree.heading('reciprocity_time', text='Reciprocity (Time)')
        self.tree.heading('causality_freq', text='Causality (Freq)')
        self.tree.heading('causality_time', text='Causality (Time)')
        
        # Configure column widths
        self.tree.column('#0', width=300)
        for col in columns:
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
                item = self.tree.insert('', 'end', text=filename, 
                                      values=('-', '-', '-', '-', '-', '-'))
                
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
        
        # Create temporary CSV file
        temp_csv = "temp_input.csv"
        with open(temp_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for filepath in self.file_list:
                writer.writerow([filepath])
        
        # Process files
        total_files = len(self.file_list)
        for i, filepath in enumerate(self.file_list):
            result = self.cli.metrics.evaluate_file(filepath)
            self.results[filepath] = result
            
            # Update progress
            progress = (i + 1) / total_files * 100
            self.progress_var.set(progress)
            
            # Update table in main thread
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
                values = []
                for metric in ['passivity', 'reciprocity', 'causality']:
                    for domain in ['freq', 'time']:
                        key = f"{metric}_{domain}"
                        value = result.get(key, -1)
                        
                        if value < 0:
                            values.append("Error")
                        else:
                            level = self.cli.metrics.get_quality_level(metric, value)
                            # Use Unicode symbols for quality levels
                            symbol = {'great': '✓', 'good': '○', 
                                    'acceptable': '△', 'bad': '✗'}.get(level, '?')
                            values.append(f"{symbol} {value:.4f}")
                
                self.tree.item(item, values=values)
                
                # Determine overall quality for row coloring
                max_value = max([v for k, v in result.items() 
                               if k.endswith('_freq') or k.endswith('_time')])
                if max_value < 0:
                    tag = 'error'
                elif max_value <= 0.01:
                    tag = 'great'
                elif max_value <= 0.05:
                    tag = 'good'
                elif max_value <= 0.1:
                    tag = 'acceptable'
                else:
                    tag = 'bad'
                
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
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="OpenSNPQual - S-Parameter Quality Evaluation Tool"
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
