# 🚀 Auto Data Cleaner Pro — Enterprise Edition

> A powerful, AI-assisted desktop application for cleaning, analyzing, and visualizing CSV/Excel datasets — with anomaly detection, quality scoring, and rich visualizations built in.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![GUI](https://img.shields.io/badge/GUI-Tkinter-orange)
![ML](https://img.shields.io/badge/ML-Isolation%20Forest-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📸 Screenshots

> _Upload a dataset and instantly see your quality score, run cleaning operations, and explore visualizations — all from one window._

---

## ✨ Features

### 🔧 Automated Data Cleaning
- **Column name standardization** — strips special characters, normalizes to `snake_case`
- **Duplicate row removal** — detects and removes exact duplicates
- **Missing value handling** — auto, drop rows, fill with median/mean
- **Data type correction** — attempts numeric coercion on ambiguous columns
- **Text cleaning** — trims whitespace, collapses repeated spaces

### 📊 Data Quality Scoring
Calculates an overall **A–F quality grade** based on four weighted metrics:

| Metric | Weight | Description |
|---|---|---|
| Completeness | 30% | % of non-null cells |
| Uniqueness | 30% | Penalizes duplicate rows |
| Consistency | 20% | Detects case variation in text columns |
| Validity | 20% | Flags negative values in age/price/quantity fields |

### 🤖 AI-Powered Anomaly Detection
Uses **Isolation Forest** (scikit-learn) on all numeric columns to identify statistically unusual rows. Reports anomaly count, percentage, and affected columns.

### 📈 Visualization Windows (each opens separately)
- **Quality Dashboard** — pie charts + bar charts for all quality metrics
- **Missing Values Heatmap** — seaborn heatmap across the full dataset
- **Correlation Matrix** — color-mapped correlation for numeric columns
- **Data Distribution** — histograms for up to 6 numeric columns
- **Data Types Pie Chart** — breakdown of dtype categories
- **Box Plots** — outlier visualization for numeric columns

### 🗂️ UI Tabs
| Tab | Purpose |
|---|---|
| 📊 Dashboard | Live quality metrics and grade |
| 🔍 Data Preview | Toggle between original and cleaned data |
| ✨ Cleaning Results | Before/after comparison |
| 📈 Visualizations | Launch all chart windows |
| 🤖 Anomaly Detection | AI anomaly report |
| 📝 Logs | Timestamped operation log |

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/auto-data-cleaner-pro.git
cd auto-data-cleaner-pro
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python main.py
```

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
chardet
tk
```

> **Note:** `tkinter` is included with most Python distributions. On Linux you may need: `sudo apt install python3-tk`

---

## 🚀 Quick Start

1. Launch the app: `python main.py`
2. Click **📂 Upload CSV/Excel File** and select your dataset
3. Review the **Dashboard** tab for an instant quality score
4. Configure cleaning options in the left panel
5. Click **▶ START CLEANING**
6. Optionally run **🔍 DETECT ANOMALIES** for AI-based row flagging
7. Use **💾 Save Cleaned Data** or **📊 Export Report** to save your results

---

## 📁 Project Structure

```
auto-data-cleaner-pro/
│
├── main.py                  # Entry point — launches the GUI
│
├── core/
│   ├── cleaner.py           # EnhancedDataCleaner — all cleaning logic
│   ├── scorer.py            # DataQualityScorer — completeness, uniqueness, etc.
│   └── anomaly.py           # AnomalyDetector — Isolation Forest wrapper
│
├── ui/
│   └── app.py               # ProfessionalDataCleaner — full Tkinter GUI
│
├── requirements.txt
└── README.md
```

> _Currently the project ships as a single-file script. The structure above reflects a recommended refactor for larger teams._

---

## ⚙️ Cleaning Configuration

| Option | Default | Description |
|---|---|---|
| Standardize Column Names | ✅ | Lowercase + snake_case |
| Remove Duplicate Rows | ✅ | Exact match deduplication |
| Handle Missing Values | ✅ | See strategy below |
| Correct Data Types | ✅ | String → numeric coercion |
| Clean Text Data | ✅ | Strip + collapse whitespace |
| Missing Strategy | `auto` | `auto` / `drop_rows` / `fill_median` / `fill_mean` |
| Missing Threshold | 50% | Columns above this % missing are dropped (`auto` mode) |

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request

Please open an issue first for major changes so we can discuss the direction.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [pandas](https://pandas.pydata.org/) — data manipulation
- [scikit-learn](https://scikit-learn.org/) — Isolation Forest anomaly detection
- [matplotlib](https://matplotlib.org/) & [seaborn](https://seaborn.pydata.org/) — visualizations
- [tkinter](https://docs.python.org/3/library/tkinter.html) — GUI framework
