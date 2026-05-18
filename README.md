🚀 Auto Data Cleaner Pro — Ultra Edition
https://python.org
LICENSE
https://pandas.pydata.org
https://scikit-learn.org
A professional desktop application for automated data cleaning, quality scoring, and interactive visualization.
<img src="https://img.shields.io/badge/GUI-Tkinter-ff69b4?logo=python" alt="GUI">
<img src="https://img.shields.io/badge/ML-Isolation%20Forest-yellow?logo=scikit-learn" alt="ML">
<img src="https://img.shields.io/badge/Charts-Matplotlib%20%7C%20Seaborn-blueviolet?logo=matplotlib" alt="Charts">
</div>
✨ Features
Table
Feature	Description
📂 Multi-format Support	Load .csv, .xlsx, and .xls files with automatic encoding detection
🔧 11 Cleaning Operations	From basic deduplication to advanced outlier removal and text normalization
📊 Real-time Quality Dashboard	Live data quality score (0–100%) with A–F grading system
🤖 AI Anomaly Detection	Isolation Forest algorithm for intelligent outlier identification
📈 8 Interactive Visualizations	Quality dashboard, heatmaps, correlation matrices, distributions, box plots, pair plots, and more
📝 Comprehensive Logging	Full audit trail of every cleaning operation with timestamps
💾 Export & Reports	Save cleaned data as CSV/Excel or generate detailed TXT reports
🖥️ Large-Font UI	Full-screen, high-contrast interface optimized for readability
🎬 Demo
plain
Copy
┌─────────────────────────────────────────────────────────────┐
│         🚀 AUTO DATA CLEANER PRO - ULTRA                    │
├──────────────────────┬──────────────────────────────────────┤
│  📁 DATA SOURCE      │  📊 DASHBOARD                        │
│  ┌────────────────┐  │  ┌──────────────────────────────┐  │
│  │ Upload CSV/    │  │  │    📈 DATA QUALITY SCORE      │  │
│  │ Excel File     │  │  │         89%                   │  │
│  └────────────────┘  │  │    Grade: B - Good Quality    │  │
│                      │  └──────────────────────────────┘  │
│  🔧 CLEANING OPTIONS │  ┌────┐ ┌────┐ ┌────┐ ┌────┐    │
│  ☑ Standardize Cols  │  │Rows│ │Cols│ │Miss│ │Dup │    │
│  ☑ Remove Duplicates │  │9.8K│ │ 14 │ │  0 │ │  0 │    │
│  ☑ Handle Missing    │  └────┘ └────┘ └────┘ └────┘    │
│  ☑ Correct Types     │                                      │
│  ☑ Clean Text        │  📈 Visualizations  🤖 Anomalies   │
│                      │                                      │
│  ▶ START CLEANING    │  🔍 Preview    📝 Logs             │
│  🔍 DETECT ANOMALIES │                                      │
│                      │                                      │
├──────────────────────┴──────────────────────────────────────┤
│  ✅ Ready — 9,847 rows × 14 columns loaded                  │
└─────────────────────────────────────────────────────────────┘
🛠️ Installation
Prerequisites
Python 3.8+
pip package manager
Step 1 — Clone the Repository
bash
Copy
git clone https://github.com/yourusername/auto-data-cleaner-pro.git
cd auto-data-cleaner-pro
Step 2 — Install Dependencies
bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
Note: openpyxl is required for Excel (.xlsx) file support.
Step 3 — Launch the Application
bash
Copy
python auto_data_cleaner.py
The app opens in full-screen mode automatically.
📖 Quick Start Guide
1️⃣ Upload Your Dataset
Click "📂 Upload CSV/Excel File" and select your data file. The app auto-detects encoding (UTF-8, Latin-1, ISO-8859-1).
2️⃣ Review the Dashboard
Switch to the 📊 Dashboard tab to see your Data Quality Score, row/column counts, missing values, duplicates, and more.
3️⃣ Configure Cleaning Options
In the left panel, toggle operations:
Table
Operation	Default	Purpose
Standardize Column Names	✅	Lowercase, underscores, remove special chars
Remove Duplicate Rows	✅	Drop identical rows
Handle Missing Values	✅	Auto-fill or drop based on strategy
Correct Data Types	✅	Auto-detect numeric/date columns
Clean Text Data	✅	Trim whitespace, normalize spaces
Remove Outliers	☐	IQR or Z-score outlier removal
Remove Constant Columns	☐	Drop single-value columns
Fix Whitespace Issues	☐	Aggressive whitespace cleaning
Normalize Text Case	☐	lower / upper / title / sentence
Remove Special Characters	☐	Strip non-alphanumeric chars
Drop High Missing Columns	☐	Remove columns >50% missing
4️⃣ Choose Strategies
Missing Strategy: auto | drop_rows | fill_median | fill_mean | fill_mode | fill_ffill | fill_bfill
Outlier Method: iqr | zscore
Text Case: lower | upper | title | sentence
5️⃣ Start Cleaning
Click "▶ START CLEANING". Review the Before/After comparison, then choose to replace or keep the original.
6️⃣ Explore & Save
Open visualizations in new maximized windows
Run anomaly detection with one click
Save cleaned data as CSV or Excel
Export a full report as TXT
📊 Visualizations
All charts open in separate, maximized windows with interactive Matplotlib toolbars (pan, zoom, save).
Table
Chart	Icon	Description
Quality Dashboard	📊	4-panel overview with score gauge, metrics bar chart, missing values, and data types
Missing Values Heatmap	🔥	Seaborn heatmap showing null value patterns
Correlation Matrix	📈	Color-coded correlation grid for numeric columns
Data Distribution	📉	Histograms for up to 6 numeric columns
Data Types Pie Chart	🥧	Proportional breakdown of column data types
Box Plots	📊	Statistical outlier visualization
Pair Plot	📊	Scatter matrix for multi-variable relationships
Scatter Matrix	📈	Pandas scatter matrix with histogram diagonals
🤖 Anomaly Detection
Powered by scikit-learn Isolation Forest:
Automatically scales numeric data with StandardScaler
Detects anomalous rows with configurable contamination (default: 10%)
Reports anomaly count, percentage, and affected columns
Provides actionable recommendations
plain
Copy
🔍 ANOMALY DETECTION RESULTS
   • Total Rows: 10,000
   • Anomaly Count: 23 rows
   • Anomaly Percentage: 0.23%
   • Affected Columns: age, price, quantity
🏗️ Architecture
plain
Copy
ProfessionalDataCleaner (Main GUI — Tkinter)
│
├── EnhancedDataCleaner (Core Engine)
│   ├── standardize_column_names()
│   ├── remove_duplicates()
│   ├── handle_missing_values()
│   ├── correct_data_types()
│   ├── clean_text()
│   ├── remove_outliers()
│   ├── remove_constant_columns()
│   ├── fix_whitespace_issues()
│   ├── normalize_text_case()
│   ├── remove_special_characters()
│   ├── drop_high_missing_columns()
│   └── clean_dataset()  ← orchestrates all operations
│
├── AnomalyDetector (AI Module — scikit-learn)
│   └── detect_anomalies()  ← Isolation Forest
│
└── DataQualityScorer (Metrics Engine)
    └── calculate_quality_score()  ← Completeness, Uniqueness, Consistency, Validity
🧹 Cleaning Operations Deep Dive
Standardize Column Names
Python
Copy
"Customer Name"      → "customer_name"
"2023 Sales"         → "col_2023_sales"
"Order Date (UTC)"   → "order_date_utc"
Handle Missing Values (Auto)
Table
Scenario	Action
Column >50% missing	Drop column
Numeric column	Fill with median
Text column	Fill with mode or "Unknown"
Remove Outliers (IQR Method)
plain
Copy
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Valid range: [Q1 - 1.5×IQR,  Q3 + 1.5×IQR]
🖥️ UI Specifications
Table
Property	Value
Default State	Full-screen / Maximized
Minimum Size	1200 × 800 px
Left Panel	400 px (scrollable)
Right Panel	Expandable (Notebook with 7 tabs)
Font Family	Segoe UI
Title Font	28px Bold
Score Font	56px Bold
Metric Font	22px Bold
Theme	Dark mode (GitHub-inspired palette)
📦 Dependencies
Table
Package	Version	Purpose
pandas	≥1.3.0	Data manipulation & I/O
numpy	≥1.21.0	Numerical computing
matplotlib	≥3.4.0	Chart rendering
seaborn	≥0.11.0	Statistical visualizations
scikit-learn	≥1.0.0	Isolation Forest anomaly detection
openpyxl	≥3.0.0	Excel file read/write
🐛 Troubleshooting
Table
Issue	Solution
Failed to load file	The app auto-tries UTF-8, Latin-1, and ISO-8859-1. If all fail, re-encode your file.
UI appears too small	Minimum resolution is 1200×800. Use full-screen mode.
Need at least 2 numeric columns	Correlation & pair plots require 2+ numeric columns. Check your data types.
Outlier removal too aggressive	Switch from IQR to Z-score, or increase the threshold.
Memory error on large files	Close other apps, or process files <500MB on 8GB RAM systems.
🗺️ Roadmap
[ ] Batch processing for multiple files
[ ] JSON and Parquet file support
[ ] Custom cleaning rule builder
[ ] Dark/Light theme toggle
[ ] Export to PDF reports
[ ] Plugin system for custom operations
🤝 Contributing
Contributions are welcome! Please follow these steps:
Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📄 License
Distributed under the MIT License. See LICENSE for more information.
🙏 Acknowledgments
Pandas — Data manipulation powerhouse
scikit-learn — Machine learning toolkit
Matplotlib & Seaborn — Visualization libraries
Tkinter — Python's standard GUI
<div align="center">
Made with ❤️ for data professionals everywhere.
⭐ Star this repo · 🐛 Report Bug · ✨ Request Feature
</div>
