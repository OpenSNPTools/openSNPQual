#!/usr/bin/env python3
"""
OpenSNPQual - S-Parameter Quality Evaluation Tool
Evaluates S-parameter quality metrics including Passivity, Reciprocity, and Causality

This is the BACKEND.
Responsibilities:

All computation and S-parameter handling:
  * P370-based evaluation for a single file (freq + time, freq-only).
  * Threshold logic (good / acceptable / inconclusive / poor).

CLI logic & report generation:
  * Process CSV list of files.
  * Export CSV + Markdown.

Version string (so both CLI and GUI can show the same version).

Example usage:
  source ~/spyder-env/bin/activate
  python3 opensnpqual.py --cli -i ./example_touchstone/example_list.csv -o test

SPDX-License-Identifier: BSD-3-Clause
"""

# Version information
OPENSNPQUAL_VERSION = "v0.1"  # Change xx to your desired version number

# IMPORTS
import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import threading
from datetime import datetime

# For S-parameter processing
import skrf as rf
from ieee370_implementation.ieee_p370_quality_freq_domain import quality_check_frequency_domain
from ieee370_implementation.ieee_p370_quality_time_domain import quality_check


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
            
            f.write("\n")
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

