#!/usr/bin/env python3
"""
OpenSNPQual - S-Parameter Quality Evaluation Tool
Evaluates S-parameter quality metrics including Passivity, Reciprocity, and Causality

This is the FRONT END / GUI. Responsibilities:

User interface:
  * Tk window, menu bar, dialogs.
  * File/folder selection.
  * Showing metrics in a TreeView, applying row coloring, etc.

Delegation to backend:
  * Create OpenSNPQualCLI from backend.
  * For each file, call cli.evaluate_file_with_time_domain() or cli.evaluate_file_frequency_only().

SPDX-License-Identifier: BSD-3-Clause
"""

# IMPORTS
import os
import sys
import csv
import argparse
from pathlib import Path

# For GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from datetime import datetime
import webbrowser
import time

# Optional drag-and-drop support
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    TkinterDnD = None
    DND_FILES = None
    DND_AVAILABLE = False

# from backend
from opensnpqual_backend import (
    OpenSNPQualCLI,
    OPENSNPQUAL_VERSION,
    OPENSNPQUAL_TITLE,
    Settings,
    load_settings,
    save_settings,
    get_settings_path,
)


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
    SNP_EXTENSIONS = ('s1p', 's2p', 's3p', 's4p', 's6p', 's8p', 'snp', 's*p')
    
    def __init__(self):
        # Use DnD-enabled root when available, fall back gracefully otherwise
        self.root = TkinterDnD.Tk() if DND_AVAILABLE else tk.Tk()
        self.root.title(OPENSNPQUAL_TITLE)
        self.root.geometry("1200x600")

        # Load persisted settings (if any)
        self.settings = load_settings()
        self.cli = OpenSNPQualCLI()
        self.parallel_enabled = self.settings.parallel_per_file  # can be flipped in future settings UI
        self.file_list = []
        self.results = {}

        self.calculate_time_domain_var = None  # Will be set in setup_ui
        
        self.setup_ui()
        self._init_drag_and_drop()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
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
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=0, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        ttk.Button(button_frame, text="Load SNPs", command=self.load_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Folder", command=self.load_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Calculate", command=self.calculate_metrics).pack(side=tk.LEFT, padx=5)
        
        # Add the checkbox for time domain calculation
        self.calculate_time_domain_var = tk.BooleanVar(value=self.settings.include_time_domain)
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
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Table frame with scrollbars
        self.table_frame = ttk.Frame(self.main_frame)
        self.table_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(3, weight=1)
        
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
            self.table_frame,
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
        vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout for table and scrollbars
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.table_frame.columnconfigure(0, weight=1)
        self.table_frame.rowconfigure(0, weight=1)
        
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
        # Persist setting
        self.settings.include_time_domain = self.calculate_time_domain_var.get()
        self._save_settings()

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
            snp_files = self._find_sparam_files(folder)
            
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
        self.root.after(0, lambda: self.status_label.config(text="Calculating..."))
        self.root.after(0, self.progress_var.set, 0)
        start_time = time.perf_counter()
        
        # Check if time domain calculation is enabled
        calculate_time_domain = self.calculate_time_domain_var.get()
        self.settings.include_time_domain = calculate_time_domain
        self.settings.parallel_per_file = self.parallel_enabled
        self._save_settings()
        filepaths = list(self.file_list)
        total_files = len(filepaths)

        settings = Settings(
            parallel_per_file=self.parallel_enabled,
            include_time_domain=calculate_time_domain,
        )

        # Progress hook executed in worker thread; UI updates are scheduled on main thread
        def _progress_hook(filepath, result, completed, total):
            self.results[filepath] = result
            progress = completed / total * 100
            self.root.after(0, self._update_table_row, filepath, result)
            self.root.after(0, self.progress_var.set, progress)

        try:
            results = self.cli.evaluate_files(
                filepaths,
                max_workers=min(os.cpu_count() or 1, total_files),
                settings=settings,
                progress_hook=_progress_hook,
            )
        except Exception as e:
            # Fallback to sequential if parallel evaluation fails
            results = []
            for i, filepath in enumerate(filepaths):
                if settings.include_time_domain:
                    result = self.cli.evaluate_file_with_time_domain(filepath)
                else:
                    result = self.cli.evaluate_file_frequency_only(filepath)
                results.append(result)
                self.results[filepath] = result
                progress = (i + 1) / total_files * 100
                self.root.after(0, self._update_table_row, filepath, result)
                self.root.after(0, self.progress_var.set, progress)
            print(f"Parallel evaluation failed, used sequential: {e}")

        # Ensure all results are tracked even if hooks missed (should not happen)
        for filepath, result in zip(filepaths, results):
            if filepath not in self.results:
                self.results[filepath] = result
                self.root.after(0, self._update_table_row, filepath, result)

        # Save results automatically
        output_csv = f"snpqual_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.cli.save_csv_results(list(self.results.values()), output_csv)
        
        output_md = output_csv.replace('.csv', '.md')
        elapsed = time.perf_counter() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        status_msg = (
            f"Finished processing {total_files} files in {minutes} min {seconds:02d} sec. "
            f"Results saved to {output_csv} and {output_md}"
        )
        self.cli.save_markdown_results(list(self.results.values()), output_md, summary=status_msg)

        self.root.after(0, lambda: self.status_label.config(text=status_msg))
    
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

    def _init_drag_and_drop(self):
        """Enable drag-and-drop of files when tkinterdnd2 is present"""
        if not DND_AVAILABLE:
            # Keep normal Ready message but hint that DnD is optional
            self.status_label.config(text="Ready (No drag-and-drop available, requires tkinterdnd2)")
            return

        # Register multiple surfaces so the first drop works anywhere in the window
        for widget in (
            self.root,
            getattr(self, "main_frame", None),
            getattr(self, "table_frame", None),
            self.tree,
        ):
            if widget:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind('<<Drop>>', self._on_drop)

        self.status_label.config(text="Ready (drag-and-drop enabled)")

    def _on_drop(self, event):
        """Handle files/folders dropped onto the window or table"""
        # splitlist handles paths with spaces wrapped in braces
        raw_paths = self.root.tk.splitlist(event.data)

        collected_files = []
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser()
            if path.is_dir():
                collected_files.extend(str(f) for f in self._find_sparam_files(path))
            elif self._is_sparam_file(path):
                collected_files.append(str(path))

        if collected_files:
            self.add_files_to_list(collected_files)
            self.status_label.config(text=f"Added {len(collected_files)} file(s) via drag-and-drop")
        else:
            self.status_label.config(text="No supported S-parameter files detected in drop")

        # Return the suggested action so tkdnd treats the drop as handled immediately
        return event.action or 'copy'

    def _is_sparam_file(self, path_obj):
        """Check if a Path or string points to a supported S-parameter file"""
        suffix = Path(path_obj).suffix.lower().lstrip('.')
        return suffix in self.SNP_EXTENSIONS

    def _find_sparam_files(self, folder):
        """Return a list of Path objects for supported files within a folder"""
        folder_path = Path(folder)
        snp_files = []
        for ext in self.SNP_EXTENSIONS:
            snp_files.extend(folder_path.glob(f"*.{ext}"))
            snp_files.extend(folder_path.glob(f"*.{ext.upper()}"))
        return snp_files
    
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

            text.insert("end", "\n\n")
            text.insert("end", "Made with ❤️ for the Signal Integrity Community", "italic")
            
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

    def _save_settings(self):
        """Persist settings to disk alongside the executable/script"""
        try:
            save_settings(self.settings)
        except Exception as e:
            print(f"Warning: failed to save settings: {e}")

    def on_close(self):
        """Handle window close"""
        self.settings.include_time_domain = self.calculate_time_domain_var.get()
        self.settings.parallel_per_file = self.parallel_enabled
        self._save_settings()
        self.root.destroy()

    def run(self):
        """Run the GUI application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description = OPENSNPQUAL_TITLE
    )
    parser.add_argument('--cli', action='store_true', 
                       help='Run in CLI mode')
    parser.add_argument('-i', '--input', type=str, 
                       help='Input CSV file with S-parameter filenames (CLI mode)')
    parser.add_argument('-o', '--output', type=str, 
                       help='Output prefix for result files (CLI mode)')
    parser.add_argument('files', nargs='*', 
                       help='S-parameter files to load (GUI mode)')
    parser.add_argument("--freq-only", action="store_true",
                    help="Run IEEE370 frequency-domain checks only (no time-domain metrics)")
    parser.add_argument("--no-parallel", action="store_true",
                    help="Disable per-file parallel processing (CLI mode)")
    
    args = parser.parse_args()
    
    if args.cli:
        # CLI mode
        if not args.input:
            print("Error: --input is required in CLI mode")
            sys.exit(1)
        
        cli = OpenSNPQualCLI()
        output_file = cli.process_csv(
            args.input,
            args.output,
            freq_only=args.freq_only,
            parallel_per_file=not args.no_parallel,
        )
        print(f"Results saved to: {output_file}")
    else:
        # GUI mode
        app = OpenSNPQualGUI()
        app.run()


if __name__ == "__main__":
    main()
