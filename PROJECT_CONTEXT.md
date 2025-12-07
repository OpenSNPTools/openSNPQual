# AI Quickstart (Read This First)

> **This section is designed for AI coding assistants such as ChatGPT, Codex, GitHub Copilot, and others.  
> Always follow these rules before generating or editing code in this repository.**

### 1. Purpose of the Project
- This project is **OpenSNPQual**, a **simple, cross-platform S-parameter quality checker**.
- It performs passivity, reciprocity, and causality checks using **IEEE P370-inspired** methods.
- It includes both a **GUI** (Tkinter) and a **CLI**.

### 2. Critical Constraints
- **Do NOT change architecture or UX without asking first.**
- **Do NOT introduce platform-specific dependencies** — this must run on Linux, macOS, and Windows.
- **Tkinter is the GUI toolkit.**  
  Do not switch to PyQt, PySide, GTK, wxPython, etc.
- **Keep the project simple** — avoid unnecessary abstractions or large frameworks.
- **Maintain IEEE 370 alignment**, especially terminology and metric definitions.

### 3. Where the Logic Lives
- `opensnpqual_backend.py` → computation + CLI  
- `opensnpqual_gui.py` → Tkinter GUI  
- `ieee_p370_quality_freq_domain.py` → P370 frequency-domain metrics  
- `ieee_p370_quality_time_domain.py` → P370 time-domain metrics  
- `testQualityCheck_2Port.py` → examples/tests  
- `README.md` → public-facing overview

### 4. How to Behave When Editing Code
- **Ask before large refactors**, interface changes, or new features.  
- **Preserve numerical integrity** — don’t change equations unless explicitly requested.  
- **Write clear comments**, especially for numerical methods (FFT, windowing, interpolation).  
- **Use small, clear functions**, not big abstractions.  
- **Propagate changes consistently**: backend ↔ GUI ↔ docs.

### 5. Metrics to Keep Consistent
- Passivity, Reciprocity, Causality (freq and time)
- Threshold categories:  
  - GOOD / ACCEPTABLE / INCONCLUSIVE / POOR  
- Return:
  - **numeric values**  
  - **string labels**  
  - **summary status**

### 6. When Unsure
> **AI assistants must ask before assuming.**  
If a question touches:
- architecture,  
- UX behavior,  
- numerical interpretation,  
- IEEE 370 semantics,  
- GUI design,  
- file loading/saving logic,

… then pause and ask for clarification.

---

# OpenSNPQual – Project Context

> **NOTE FOR AI ASSISTANTS (Codex / ChatGPT / etc.):**  
> This file describes the goals, architecture, and constraints of the OpenSNPQual project.  
> **Always follow the goals and constraints in this document when editing code.**

---

## 1. What OpenSNPQual Is

OpenSNPQual is a **simple, cross-platform S-parameter (Touchstone) quality checker**.

**Primary purpose:**

- Load Touchstone `.sNp` files (2-port and N-port).
- Evaluate **S-parameter quality** using **IEEE P370**-inspired methods:
  - **Passivity**
  - **Reciprocity**
  - **Causality**
- Provide **frequency-domain and time-domain** quality metrics.
- Present results as:
  - **A simple, immediately readable report** (CLI + exported CSV/Markdown).
  - **A lightweight GUI** (Tkinter) for interactive use.

This project is intentionally **educational and exploratory**:
- It re-implements / adapts IEEE 370 reference code in Python.
- It is *not* guaranteed to match the original reference code bit-for-bit.
- Users should treat it as a **learning tool** and **sanity checker**, not as a formally certified sign-off tool.

---

## 2. High-Level Goals and Non-Goals

### 2.1 Goals

1. **Cross-platform**  
   - Must run on **Windows, Linux, and macOS** using standard Python + common libraries.
   - Avoid platform-specific GUIs or shell logic where possible.

2. **Simplicity & usability**
   - Easy for an SI/PI engineer to install and run.
   - GUI that feels simple and obvious:
     - Add files
     - Run quality checks
     - View results in a table
   - CLI that can process **multiple Touchstone files** from a list or directory.

3. **IEEE P370 correlation**
   - Stay conceptually aligned with **IEEE P370-2020** quality checks.
   - Clearly distinguish **frequency-domain** vs **time-domain** metrics.
   - Respect P370 concepts & nomenclature as much as practical.

4. **Transparent, quantitative metrics**
   - Provide:
     - **Quantitative scalar metrics** (e.g., % passivity, % reciprocity, % causality, time-domain mV metrics).
     - **Qualitative levels** (e.g., `good`, `acceptable`, `inconclusive`, `poor`).
   - Make it easy to compare results between channels and between files.

5. **Minimal dependencies**
   - Stick to Python standard library + NumPy / SciPy / Matplotlib / scikit-rf (if used) and Tkinter.
   - No heavy frameworks; avoid complex build systems.

6. **Easy to extend**
   - New metrics, additional plotting, or batch/report workflows should be straightforward to add.
   - Back-end computation logic separated from GUI/CLI.

### 2.2 Non-Goals

- Not a full **channel compliance** tool.
- Not aiming for 1:1 **golden correlation** to every nuance of IEEE 370 reference code.
- Not a replacement for full commercial SI/PI toolchains.
- Not intended to be a heavy GUI suite; must remain “small and friendly”.

---

## 3. Repository Structure (Key Files)

> AI assistants: please **respect this structure** and update sections consistently when you add new files.

- `opensnpqual.py`  
  - Tiny entry point that imports and calls `main()` from `opensnpqual_gui`.
  - Allows launching the GUI as a script.

- `opensnpqual_gui.py`  
  - **Tkinter-based GUI** front-end.
  - Responsibilities:
    - Create the main window, menus, file list, and results table.
    - Handle user interactions:
      - Add / remove Touchstone files.
      - Run quality checks on selected/all files.
      - Show **IEEE 370 correlation** info (Help menu).
      - Provide “About OpenSNPQual” dialog.
      - Provide simple “Report a bug” link/action.
    - Display **quality metrics** and statuses in a table:
      - Per file: passivity, reciprocity, causality, etc.
      - Color coding / severity where appropriate.
    - Call the back-end functions to perform actual analysis.

- `opensnpqual_backend.py`  
  - **Core computation & CLI logic**.
  - Responsibilities:
    - Parse and handle S-parameter data (typically via helper functions or scikit-rf, depending on implementation).
    - Implement **IEEE P370-inspired** quality checks in both **frequency** and **time** domains.
    - Provide a central class (e.g. `SParameterQualityMetrics`) that:
      - Holds threshold definitions.
      - Computes numeric metrics.
      - Classifies metrics into **good / acceptable / inconclusive / poor**.
    - Implement report generation:
      - Command-line interface (CLI) for batch processing.
      - Export results to CSV and Markdown.
    - Provide a **single version string** used by both CLI and GUI.

- `ieee_p370_quality_freq_domain.py`  
  - Python adaptation of the IEEE 370 **frequency-domain quality check** code.
  - Adapted from original Octave/MATLAB reference implementation:
    - `qualityCheckFrequencyDomain.m` and related scripts.
  - Computes frequency-domain quality measures, e.g.:
    - Passivity quality metrics (PQMi).
    - Reciprocity quality metrics (RQMi).
    - Causality frequency metrics related to P370.

- `ieee_p370_quality_time_domain.py`  
  - Python adaptation of the IEEE 370 **time-domain quality check** code.
  - Also derived from the official IEEE 370 open-source reference.
  - Responsible for:
    - Converting S-parameters to time domain (TDR/TDT-like responses).
    - Computing time-domain deviations for passivity/reciprocity/causality.
    - Producing mV-level metrics and/or time-domain quality measures (PQMa, RQMa, CQMa).

- `testQualityCheck_2Port.py`  
  - Example / test script for **2-port S-parameter** quality checks.
  - Demonstrates:
    - Reading a Touchstone file.
    - Calling the quality check functions.
    - Printing or plotting results for verification.

- `IEEE_370-2020.pdf`  
  - Local copy of the **IEEE 370-2020 standard** document.
  - Used as a reference to understand:
    - Exact definitions of passivity/reciprocity/causality metrics.
    - Recommended processes for frequency and time domain evaluations.
  - Not distributed in some contexts; usage must respect IEEE licensing.

- `README.md`  
  - Public-facing description (GitHub-ready).
  - Summarizes:
    - What OpenSNPQual does.
    - How to install and run it.
    - Screenshot of the GUI.
    - Licensing (BSD 3-Clause).
    - Disclaimer that this is a **learning / exploratory** codebase.

- `docs/` (if present)
  - Screenshots, additional documentation, or examples.

---

## 4. S-Parameter Quality Metrics

### 4.1 Conceptual definitions

**Passivity**  
- A passive network should not generate net power.
- In S-parameter terms, the power gain should be ≤ 1 for all ports and frequencies.
- The tool estimates how well the provided S-parameters satisfy passivity.

**Reciprocity**  
- For reciprocal networks `S_ij = S_ji`.
- The tool measures how close the S-parameter matrix is to being symmetric (within numerical error and measurement noise).

**Causality**  
- Causal systems produce zero response before the stimulus.
- In the frequency domain, this is related to the Kramers-Kronig / Hilbert-transform-like relationships between real and imaginary parts.
- In the time domain, we look at whether the impulse/step response has energy **before** t=0 or exhibits non-physical behavior.

### 4.2 Frequency-domain metrics (PQMi / RQMi / CQMi)

The **frequency-domain** quality metrics are typically expressed as **percentages**, where:

- `0%` → completely fails the criterion.
- `100%` → perfectly satisfies the criterion, within numerical tolerance.

OpenSNPQual defines thresholds like (example, subject to refinement in code):

- **Passivity / Reciprocity** (% of data points meeting the requirement):
  - GOOD:        (99.9, 100]
  - ACCEPTABLE:  (99.0, 99.9]
  - INCONCLUSIVE:(80.0, 99.0]
  - POOR:        [0, 80.0]

- **Causality** (percentage-based measure):
  - GOOD:        (80.0, 100]
  - ACCEPTABLE:  (50.0, 80.0]
  - INCONCLUSIVE:(20.0, 50.0]
  - POOR:        [0, 20.0]

> AI assistants: when modifying thresholds, **keep the logic self-consistent** and synchronized between documentation, code comments, and GUI labels.

### 4.3 Time-domain metrics (PQMa / RQMa / CQMa in mV)

The **time-domain** quality metrics are typically expressed in **mV deviations**:

- Lower is better (less violation / less non-physical behavior).

Typical threshold scheme (example):

- GOOD:        [0, 5)   mV
- ACCEPTABLE:  [5, 10)  mV
- INCONCLUSIVE:[10, 15) mV
- POOR:        [15, ∞)  mV

The back-end provides helper methods to:

- Compute these mV-level metrics from time-domain responses.
- Classify them into good/acceptable/inconclusive/poor.
- Return both numeric values (floats) and labels (strings).

---

## 5. GUI Behaviour and UX Expectations

### 5.1 Main workflow

Typical user workflow:

1. Start the GUI (e.g., `python -m opensnpqual` or similar).
2. Add one or more `.sNp` Touchstone files:
   - Via “File → Add…” or an “Add Files” button.
3. For each file, the GUI:
   - Shows filename, path, number of ports.
   - Offers a way to trigger calculation (button or menu).
4. On “Run Quality Check”:
   - Calls back-end computation for each selected file.
   - Displays metrics in a **table** (treeview) with columns like:
     - File name
     - Passivity (freq)
     - Reciprocity (freq)
     - Causality (freq)
     - Passivity (time)
     - Reciprocity (time)
     - Causality (time)
     - Overall status / summary.
5. Optionally allows:
   - Copying table to clipboard.
   - Clearing all entries.
   - Viewing correlation notes to IEEE 370.
   - Opening an “About” dialog and “Report a bug”.

### 5.2 UX principles

- **Do not overwhelm the user with options.**
- Provide **clear labels** and **tooltips**, but keep screens uncluttered.
- Failures (e.g., file load errors) should produce **dialog boxes** or status messages that:
  - Are concise.
  - Suggest what the user can try next.

---

## 6. Coding Style and Conventions

### 6.1 General Python style

- Prefer **PEP 8** style (snake_case for functions, CamelCase for classes).
- Type hints where reasonable, especially public APIs.
- Docstrings for public functions and classes describing:
  - Inputs
  - Outputs
  - Units (e.g., Hz, GHz, m, ps, mV)
- Keep functions focused; avoid overly long functions doing too many things.

### 6.2 Numerical work

- Use **NumPy** for vectorized operations.
- Be explicit about units:
  - Frequency arrays in **Hz** (or document if in GHz).
  - Data rate in **Gbps** but convert consistently where required.
  - Time domain arrays in **seconds** or **ps**; document clearly.
- Be mindful of:
  - Padding / extrapolation.
  - Windowing.
  - FFT/IFFT normalization factors.
  - Stability and conditioning (e.g., avoid dividing by extremely small values without checks).

### 6.3 Error handling

- For GUI:
  - Catch exceptions and show a dialog instead of crashing.
- For CLI:
  - Exit with non-zero code on error.
  - Print a clear message to stderr.

### 6.4 Testing and examples

- Keep small example scripts like `testQualityCheck_2Port.py`:
  - They serve both as smoke tests and documentation.
- When adding new features:
  - Try to add at least a minimal reproducible script demonstrating usage.

---

## 7. Licensing and Attribution

- Project is under **BSD 3-Clause** license.
- Adapted IEEE 370 reference code:
  - Must include attribution to original IEEE 370 open source authors.
  - Maintain any required headers in the adapted files.
- AI assistants: when generating new code derived from IEEE 370 algorithms:
  - **Do not remove** existing copyright and attribution headers.
  - If you create new files that are clearly adaptations of IEEE 370 reference scripts, mirror attribution appropriately.

---

## 8. Future / Planned Directions (Informative)

> These are **ideas**, not strict requirements. AI assistants should treat them as **nice-to-have** and only implement when explicitly requested.

- Integration with other OpenSNPTools utilities (e.g., plotting / publishing).
- Additional metrics:
  - Insertion loss deviation (ILD), FOMILD, skew, etc.
- Better batch processing GUI (tree/table with progress, filters, summaries).
- Export to more report formats (Markdown → PDF, HTML, etc).

---

## 9. Instructions for AI Assistants (Codex / ChatGPT)

**You are assisting with OpenSNPQual development. Please:**

1. **Respect cross-platform constraints**
   - Do *not* introduce Windows-only or macOS-only dependencies.
   - Tkinter must remain the GUI toolkit; avoid switching toolkits unless explicitly requested.

2. **Do not silently change thresholds or semantics**
   - If you think thresholds or classification scheme should change,:
     - Explain the reasoning.
     - Update this `PROJECT_CONTEXT.md`, the code, and any user-facing text consistently.

3. **Preserve numerical integrity**
   - If you refactor numerical routines:
     - Validate against existing scripts (e.g., `testQualityCheck_2Port.py`) or add new tests.
   - Keep units consistent and documented.

4. **Keep things simple**
   - Avoid overengineering.
   - Prefer small, clear functions over complex abstractions.

5. **Keep educational value**
   - Add **helpful comments** when implementing complex numerical steps (FFTs, extrapolation, windowing, etc.).
   - When porting from IEEE reference code, explain mapping in comments when non-obvious.

6. **When in doubt, ask (for human or higher-level clarification)**
   - If a change impacts architecture, UX, or numerical methodology, treat it as a design decision, not a trivial edit.

---

_End of `PROJECT_CONTEXT.md`._
