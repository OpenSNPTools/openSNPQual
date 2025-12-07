[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com/OpenSNPTools/OpenSNPQual)

# openSNPQual V0.1:  A Simple S-Parameter Quality Checker

A simple utility tool for evaluating S-parameter (Touchstone) file quality based on IEEE 370 standard. OpenSNPQual provides both GUI and CLI interfaces to assess passivity, reciprocity, and causality metrics in frequency and time domains.

Based on IEEE 370 code, re-implemented in Python using AI coders.

**This code and repo is intended to be a learning experience. The results are correlated to the original code. Use discretion and caution when using. **

![OpenSNPQual Screenshot GUI](./docs/graphics/screenshot_gui.png)

![OpenSNPQual Screenshot Result](./docs/graphics/screenshot_result_md.png)

## 🚀 Features

- **Comprehensive S-parameter Analysis**
  - Passivity verification (|S| ≤ 1)
  - Reciprocity check (S_ij = S_ji)
  - Causality evaluation using Kramers-Kronig relations
  - Python implementaion of IEEE370 code -- correlation [report here](./docs/openSNPQual_correlation_IEEE370.md)  
  
- **Dual Domain Analysis**
  - Frequency domain metrics
  - Time domain validation
  
- **Multiple Interfaces**
  - User-friendly GUI with drag-and-drop support
  - Command-line interface for automation
  - Windows "Open with" integration
  
- **Rich Output Formats**
  - CSV export for data processing
  - Markdown reports with color-coded quality indicators
  - Clipboard support for Excel/PowerPoint integration

### 📊 Quality Metrics - Initial (Frequency Domain) - good for quick check

| Level | Symbol | Passivity (PQMi) | Reciprocity (RQMi)  | Causality (CQMi) | Description |
|-------|--------|-----------|-----------|-----------|-------------|
| 🟢 Good | ✓ | (99.9, 100] | (99.9, 100]  | (80, 100] | Excellent quality, suitable for critical applications |
| 🔵 Acceptable | ○ | (99, 99.9] | (99, 99.9] | (50, 80] | OK quality, may not be suitable for sensitive applications like de-embedding |
| 🟡 Inconclusive | △ | (80, 99] | (80, 99] | (20, 50] | Marginal quality, unlikely to be reliable |
| 🔴 POOR | ✗ | [0, 80] | [0, 80] | [0, 20] | Poor quality, do not use! Re-measurement (+ VNA recalibration) recommended |

### 📊 Quality Metrics - Application-based (Time Domain) - rigorously computed

| Level | Symbol | Passivity (PQMa) | Reciprocity (RQMa)  | Causality (CQMa) | Description |
|-------|--------|-----------|-----------|-----------|-------------|
| 🟢 Good | ✓ | [0 mV, 5 mV) | [0 mV, 5 mV) | [0 mV, 5 mV) | Excellent quality, suitable for critical applications |
| 🔵 Acceptable | ○ | [5 mV, 10 mV) | [5 mV, 10 mV) | [5 mV, 10 mV) | OK quality, may not be suitable for sensitive applications like de-embedding |
| 🟡 Inconclusive | △ | [10 mV, 15 mV) | [10 mV, 15 mV) | [10 mV, 15 mV) | Marginal quality, unlikely to be reliable |
| 🔴 POOR | ✗ | [15 mV, +∞) | [15 mV, +∞) | [15 mV, +∞) | Poor quality, do not use! Re-measurement (+ VNA recalibration) recommended |

Reference:"[IEEE Standard for Electrical Characterization of Printed Circuit Board and Related Interconnects at Frequencies up to 50 GHz,](https://ieeexplore.ieee.org/document/9316329/)" in IEEE Std 370-2020 , vol., no., pp.1-147, 8 Jan. 2021, doi: 10.1109/IEEESTD.2021.9316329. 


## 🔧 Installation

### Option 1: From Source

```bash
# Clone the repository
git clone https://github.com/OpenSNPTools/openSNPQual.git
cd OpenSNPQual

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python opensnpqual.py
```


## 📖 Usage

### GUI Mode

Launch the graphical interface:

```bash
# Using executable
OpenSNPQual.exe

# From source
python opensnpqual.py
```

**GUI Features:**
- **Load SNP Files**: Click to select one or multiple S-parameter files
- **Load Folder**: Process all S-parameter files in a directory
- **Calculate**: Analyze loaded files and display results
- **Copy Table**: Copy results to clipboard for Excel/PowerPoint
- **Export**: Save results as CSV or Markdown

### CLI Mode

Process files from command line:

```bash
# Basic usage
opensnpqual --cli -i input_files.csv

# With custom output prefix
opensnpqual --cli -i input_files.csv -o my_results

# Process specific files directly
opensnpqual --cli file1.s2p file2.s4p file3.s8p

# To run on examplef files included in the repo
python3 opensnpqual.py --cli -i ./example_touchstone/example_list.csv -o test

# To run same, but only FREQ domain (initial) metrics
python3 opensnpqual.py --cli --freq-only -i ./example_touchstone/example_list.csv -o test
```

**Input CSV Format:**
```csv
path/to/file1.s2p
path/to/file2.s4p
path/to/file3.s8p
```

**Output Files:**
- `{prefix}_result.csv` - Numerical results
- `{prefix}_result.md` - Formatted report with quality indicators

### Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Signal processing
- `scikit-rf` - S-parameter file handling
- `tkinter` - GUI framework (included with Python)
- `tkinterdnd2` - Drag-and-drop support for the GUI (optional; GUI still works without it)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🔗 Related Tools

- [scikit-rf](https://scikit-rf.org/) - RF/Microwave engineering toolkit
- [IEEE 370 MATLAB Repository](https://opensource.ieee.org/elec-char/ieee-370/) - Reference implementation
- [Simbeor](https://www.simberian.com/) - Commercial S-parameter analysis
- [Keysight PLTS](https://www.keysight.com/us/en/product/N1930B/physical-layer-test-system-plts-software.html) - Physical layer test system

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IEEE P370 Working Group for quality metric standards
- scikit-rf developers for the excellent S-parameter library
- Contributors and testers from the signal integrity community

## 📧 Contact

- **Author**: Giorgi Maghlakelidze
- **Email**: giorgi.snp [@] pm [DOT] me
- **LinkedIn**: [my LinkedIn profile](https://linkedin.com/in/giorgim)
- **Bugs & Feature Requests**: [GitHub Issues](https://github.com/OpenSNPTools/openSNPQual/issues)

---

**Note**: This tool is intended for educational use. While we strive for accuracy, please always validate critical results with established professional tools.

<p align="center">Made with ❤️ for the Signal Integrity Community</p>
