"""
Auto Data Cleaner Pro - Enhanced Edition
With Multiple Visualization Windows & Better UI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime
import re
import chardet
import os
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ================================
# ANOMALY DETECTION CLASS
# ================================

class AnomalyDetector:
    """AI-based anomaly detection using Isolation Forest"""

    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()

    def detect_anomalies(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {'anomalies': [], 'anomaly_scores': [], 'anomaly_count': 0,
                    'anomaly_percentage': 0, 'affected_columns': []}

        X = df[numeric_cols].fillna(df[numeric_cols].median())
        X_scaled = self.scaler.fit_transform(X)
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        anomalies = np.where(predictions == -1)[0].tolist()

        return {
            'anomalies': anomalies,
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0,
            'affected_columns': list(numeric_cols)
        }


# ================================
# QUALITY SCORER CLASS
# ================================

class DataQualityScorer:
    """Calculate data quality metrics"""

    def calculate_quality_score(self, df):
        if df is None or len(df) == 0:
            return {'total_score': 0, 'grade': 'F', 'description': 'No data',
                    'metrics': {'completeness': 0, 'uniqueness': 0, 'consistency': 0, 'validity': 0}}

        completeness = self._completeness_score(df)
        uniqueness = self._uniqueness_score(df)
        consistency = self._consistency_score(df)
        validity = self._validity_score(df)

        total_score = (completeness * 0.3 + uniqueness * 0.3 + consistency * 0.2 + validity * 0.2)

        if total_score >= 90:
            grade = 'A'
            description = 'Excellent data quality'
        elif total_score >= 75:
            grade = 'B'
            description = 'Good data quality'
        elif total_score >= 60:
            grade = 'C'
            description = 'Fair data quality'
        elif total_score >= 40:
            grade = 'D'
            description = 'Poor data quality'
        else:
            grade = 'F'
            description = 'Very poor data quality'

        return {
            'total_score': round(total_score, 2),
            'grade': grade,
            'description': description,
            'metrics': {
                'completeness': round(completeness, 2),
                'uniqueness': round(uniqueness, 2),
                'consistency': round(consistency, 2),
                'validity': round(validity, 2)
            }
        }

    def _completeness_score(self, df):
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 0
        filled_cells = total_cells - df.isnull().sum().sum()
        return (filled_cells / total_cells) * 100

    def _uniqueness_score(self, df):
        if len(df) == 0:
            return 100
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        return max(0, 100 - duplicate_percentage)

    def _consistency_score(self, df):
        if len(df.columns) == 0:
            return 100
        consistency_issues = 0
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 0:
                    case_variations = len(set([str(v).lower() for v in unique_values]))
                    if case_variations < len(unique_values):
                        consistency_issues += 1
        return max(0, 100 - (consistency_issues / len(df.columns) * 20))

    def _validity_score(self, df):
        invalid_values = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'age' in col.lower() or 'price' in col.lower() or 'quantity' in col.lower():
                invalid_values += (df[col] < 0).sum()
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 100
        invalid_percentage = (invalid_values / total_cells) * 100
        return max(0, 100 - invalid_percentage * 2)


# ================================
# ENHANCED DATA CLEANING CORE
# ================================

class EnhancedDataCleaner:
    """Professional data cleaning engine"""

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.cleaning_log = []
        self.original_shape = None
        self.cleaned_shape = None
        self.cleaning_summary = {}

    def log(self, operation, details, level='INFO'):
        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {operation}: {details}"
        self.cleaning_log.append(log_entry)
        if self.log_callback:
            self.log_callback(log_entry)

    def load_dataset(self, file_path, file_type='csv'):
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
            self.original_shape = df.shape
            self.log("Load Dataset", f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns", 'SUCCESS')
            return df
        except Exception as e:
            self.log("Load Dataset", f"Failed: {str(e)}", 'ERROR')
            raise

    def standardize_column_names(self, df):
        original_cols = df.columns.tolist()
        def standardize(name):
            name = str(name).strip().lower()
            name = re.sub(r'[^\w\s]', '', name)
            name = re.sub(r'\s+', '_', name)
            name = re.sub(r'_+', '_', name)
            name = name.strip('_')
            if name and name[0].isdigit():
                name = f'col_{name}'
            return name or 'column'

        new_columns = [standardize(col) for col in df.columns]
        seen = {}
        for i, col in enumerate(new_columns):
            if col in seen:
                seen[col] += 1
                new_columns[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0

        df.columns = new_columns
        changes = sum(1 for o, n in zip(original_cols, new_columns) if o != n)
        self.log("Column Standardization", f"Renamed {changes} columns", 'SUCCESS')
        self.cleaning_summary['columns_renamed'] = changes
        return df

    def remove_duplicates(self, df):
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        self.log("Duplicate Removal", f"Removed {duplicates_removed:,} duplicate rows", 'SUCCESS')
        self.cleaning_summary['duplicates_removed'] = duplicates_removed
        return df

    def handle_missing_values(self, df, strategy='auto', threshold=50):
        missing_before = df.isnull().sum().sum()
        columns_dropped = []

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
            missing_pct = (missing_count / len(df)) * 100

            if strategy == 'auto':
                if missing_pct > threshold:
                    df = df.drop(columns=[col])
                    columns_dropped.append(col)
                    self.log("Missing Values", f"Dropped column '{col}' ({missing_pct:.1f}% missing)", 'INFO')
                elif df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_val = df[col].mode()
                    df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown', inplace=True)
            elif strategy == 'drop_rows':
                df = df.dropna()
                break
            elif strategy in ['fill_median', 'fill_mean'] and df[col].dtype in ['int64', 'float64']:
                fill_val = df[col].median() if strategy == 'fill_median' else df[col].mean()
                df[col].fillna(fill_val, inplace=True)

        missing_after = df.isnull().sum().sum()
        self.log("Missing Values", f"Reduced from {missing_before:,} to {missing_after:,} missing values", 'SUCCESS')
        self.cleaning_summary['missing_before'] = missing_before
        self.cleaning_summary['missing_after'] = missing_after
        self.cleaning_summary['columns_dropped'] = columns_dropped
        return df

    def correct_data_types(self, df):
        type_changes = []
        for col in df.columns:
            original_type = df[col].dtype
            if df[col].dtype == 'object':
                try:
                    numeric_attempt = pd.to_numeric(df[col], errors='coerce')
                    if numeric_attempt.notna().sum() > len(df) * 0.7:
                        df[col] = numeric_attempt
                        type_changes.append(col)
                except:
                    pass
        self.log("Data Type Correction", f"Converted {len(type_changes)} columns", 'SUCCESS')
        self.cleaning_summary['type_changes'] = len(type_changes)
        return df

    def clean_text(self, df):
        cleaned = 0
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            cleaned += 1
        self.log("Text Cleaning", f"Cleaned {cleaned} text columns", 'SUCCESS')
        return df

    def clean_dataset(self, df, config):
        cleaned_df = df.copy()
        cleaning_operations = []

        if config.get('standardize_columns', True):
            cleaned_df = self.standardize_column_names(cleaned_df)
            cleaning_operations.append('Column Standardization')

        if config.get('remove_duplicates', True):
            cleaned_df = self.remove_duplicates(cleaned_df)
            cleaning_operations.append('Duplicate Removal')

        if config.get('handle_missing', True):
            cleaned_df = self.handle_missing_values(cleaned_df, config.get('missing_strategy', 'auto'),
                                                    config.get('missing_threshold', 50))
            cleaning_operations.append('Missing Value Handling')

        if config.get('correct_types', True):
            cleaned_df = self.correct_data_types(cleaned_df)
            cleaning_operations.append('Data Type Correction')

        if config.get('clean_text', True):
            cleaned_df = self.clean_text(cleaned_df)
            cleaning_operations.append('Text Cleaning')

        self.cleaned_shape = cleaned_df.shape
        self.cleaning_summary['cleaning_operations'] = cleaning_operations
        self.cleaning_summary['original_shape'] = self.original_shape
        self.cleaning_summary['cleaned_shape'] = self.cleaned_shape

        return cleaned_df


# ================================
# MAIN GUI APPLICATION
# ================================

class ProfessionalDataCleaner:
    """Professional GUI with improved visualization"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Auto Data Cleaner Pro - Enterprise Edition")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1a1a2e')

        # Components
        self.data_cleaner = None
        self.anomaly_detector = AnomalyDetector()
        self.quality_scorer = DataQualityScorer()

        # State
        self.current_dataset = None
        self.cleaned_dataset = None
        self.original_dataset = None
        self.current_file_path = None

        # Color scheme
        self.colors = {
            'primary': '#0f3460',
            'secondary': '#16213e',
            'accent': '#e94560',
            'success': '#4ecdc4',
            'warning': '#ff6b6b',
            'text': '#eeeeee',
            'text_light': '#cccccc',
            'card': '#1a1a2e',
            'card_dark': '#0f0f1a'
        }

        self.setup_ui()

    def setup_ui(self):
        """Setup main UI"""
        # Top Header
        header = tk.Frame(self.root, bg=self.colors['primary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(header, text="🚀 AUTO DATA CLEANER PRO",
                        font=("Arial", 20, "bold"), bg=self.colors['primary'], fg='white')
        title.pack(pady=15)

        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['secondary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Panel
        left_panel = tk.Frame(main_frame, bg=self.colors['card'], width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right Panel (Notebook)
        right_panel = tk.Frame(main_frame, bg=self.colors['secondary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Setup left panel
        self.setup_left_panel(left_panel)

        # Setup notebook
        self.setup_notebook(right_panel)

        # Status Bar
        status_bar = tk.Frame(self.root, bg=self.colors['primary'], height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(status_bar, text="✅ Ready",
                                     bg=self.colors['primary'], fg='white', anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(status_bar, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=10)

    def setup_left_panel(self, parent):
        """Setup left panel with controls"""
        # Upload Section
        upload_frame = tk.LabelFrame(parent, text="📁 DATA SOURCE",
                                     font=("Arial", 12, "bold"), bg=self.colors['card'],
                                     fg=self.colors['text'], padx=10, pady=10)
        upload_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(upload_frame, text="📂 Upload CSV/Excel File",
                 command=self.upload_file, bg=self.colors['accent'], fg='white',
                 font=("Arial", 10, "bold"), height=2).pack(fill=tk.X, pady=5)

        self.file_info = tk.Text(upload_frame, height=6, bg=self.colors['card_dark'],
                                  fg=self.colors['text_light'], font=("Arial", 9), wrap=tk.WORD)
        self.file_info.pack(fill=tk.X, pady=5)

        # Cleaning Options
        clean_frame = tk.LabelFrame(parent, text="🔧 CLEANING OPTIONS",
                                    font=("Arial", 12, "bold"), bg=self.colors['card'],
                                    fg=self.colors['text'], padx=10, pady=10)
        clean_frame.pack(fill=tk.X, padx=10, pady=10)

        self.clean_vars = {}
        options = [
            ("✓ Standardize Column Names", True),
            ("✓ Remove Duplicate Rows", True),
            ("✓ Handle Missing Values", True),
            ("✓ Correct Data Types", True),
            ("✓ Clean Text Data", True)
        ]

        for text, default in options:
            var = tk.BooleanVar(value=default)
            cb = tk.Checkbutton(clean_frame, text=text, variable=var,
                               bg=self.colors['card'], fg=self.colors['text'],
                               selectcolor=self.colors['card'], font=("Arial", 10))
            cb.pack(anchor='w', pady=2)
            self.clean_vars[text] = var

        # Strategy
        strategy_frame = tk.Frame(clean_frame, bg=self.colors['card'])
        strategy_frame.pack(fill=tk.X, pady=10)

        tk.Label(strategy_frame, text="Missing Strategy:", bg=self.colors['card'],
                fg=self.colors['text']).pack(side=tk.LEFT)
        self.missing_strategy = ttk.Combobox(strategy_frame, values=['auto', 'drop_rows', 'fill_median', 'fill_mean'], width=15)
        self.missing_strategy.pack(side=tk.RIGHT)
        self.missing_strategy.set('auto')

        # Action Buttons
        tk.Button(clean_frame, text="▶ START CLEANING", command=self.start_cleaning,
                 bg=self.colors['success'], fg='white', font=("Arial", 11, "bold"),
                 height=2).pack(fill=tk.X, pady=5)

        tk.Button(clean_frame, text="🔍 DETECT ANOMALIES", command=self.detect_anomalies,
                 bg=self.colors['warning'], fg='white', font=("Arial", 10, "bold"),
                 height=1).pack(fill=tk.X, pady=5)

        # Save Section
        save_frame = tk.LabelFrame(parent, text="💾 SAVE DATA",
                                   font=("Arial", 12, "bold"), bg=self.colors['card'],
                                   fg=self.colors['text'], padx=10, pady=10)
        save_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(save_frame, text="💾 Save Cleaned Data", command=self.save_cleaned_data,
                 bg=self.colors['primary'], fg='white', font=("Arial", 10)).pack(fill=tk.X, pady=2)

        tk.Button(save_frame, text="📊 Export Report", command=self.export_report,
                 bg=self.colors['primary'], fg='white', font=("Arial", 10)).pack(fill=tk.X, pady=2)

    def setup_notebook(self, parent):
        """Setup notebook tabs"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs
        self.dashboard_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.preview_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.clean_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.viz_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.anomaly_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])
        self.logs_tab = tk.Frame(self.notebook, bg=self.colors['secondary'])

        self.notebook.add(self.dashboard_tab, text="📊 Dashboard")
        self.notebook.add(self.preview_tab, text="🔍 Data Preview")
        self.notebook.add(self.clean_tab, text="✨ Cleaning Results")
        self.notebook.add(self.viz_tab, text="📈 Visualizations")
        self.notebook.add(self.anomaly_tab, text="🤖 Anomaly Detection")
        self.notebook.add(self.logs_tab, text="📝 Logs")

        self.setup_dashboard_tab()
        self.setup_preview_tab()
        self.setup_clean_tab()
        self.setup_viz_tab()
        self.setup_anomaly_tab()
        self.setup_logs_tab()

    def setup_dashboard_tab(self):
        """Setup colorful dashboard"""
        # Welcome frame
        welcome_frame = tk.Frame(self.dashboard_tab, bg=self.colors['secondary'])
        welcome_frame.pack(fill=tk.X, padx=20, pady=20)

        tk.Label(welcome_frame, text="📊 DATA QUALITY DASHBOARD",
                font=("Arial", 18, "bold"), bg=self.colors['secondary'],
                fg=self.colors['accent']).pack()

        tk.Label(welcome_frame, text="Comprehensive data quality metrics at a glance",
                font=("Arial", 11), bg=self.colors['secondary'], fg=self.colors['text_light']).pack()

        # Metrics grid
        self.metrics_frame = tk.Frame(self.dashboard_tab, bg=self.colors['secondary'])
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Quality Score Card
        self.quality_card = tk.Frame(self.metrics_frame, bg=self.colors['card'], relief=tk.RAISED, bd=2)
        self.quality_card.pack(fill=tk.X, pady=10)

        tk.Label(self.quality_card, text="📈 DATA QUALITY SCORE",
                font=("Arial", 14, "bold"), bg=self.colors['card'], fg=self.colors['text']).pack(pady=10)

        self.score_label = tk.Label(self.quality_card, text="--",
                                    font=("Arial", 48, "bold"), bg=self.colors['card'], fg=self.colors['accent'])
        self.score_label.pack(pady=10)

        self.grade_label = tk.Label(self.quality_card, text="--",
                                    font=("Arial", 16), bg=self.colors['card'], fg=self.colors['success'])
        self.grade_label.pack(pady=5)

        # Metrics cards grid
        metrics_grid = tk.Frame(self.metrics_frame, bg=self.colors['secondary'])
        metrics_grid.pack(fill=tk.BOTH, expand=True, pady=10)

        metrics = [
            ("📊 Total Rows", "rows_label", "#4ecdc4"),
            ("📋 Total Columns", "cols_label", "#4ecdc4"),
            ("⚠️ Missing Values", "missing_label", "#ff6b6b"),
            ("🔄 Duplicate Rows", "duplicate_label", "#ff6b6b"),
            ("✅ Completeness", "complete_label", "#4ecdc4"),
            ("🎯 Uniqueness", "unique_label", "#4ecdc4"),
            ("💾 Memory Usage", "memory_label", "#f9ca24"),
            ("🔧 Data Types", "dtypes_label", "#f9ca24")
        ]

        self.metric_labels = {}

        for i, (name, key, color) in enumerate(metrics):
            card = tk.Frame(metrics_grid, bg=self.colors['card'], relief=tk.RAISED, bd=1)
            card.grid(row=i//4, column=i%4, padx=10, pady=10, sticky='nsew')

            tk.Label(card, text=name, font=("Arial", 10), bg=self.colors['card'],
                    fg=self.colors['text_light']).pack(pady=5)
            label = tk.Label(card, text="--", font=("Arial", 16, "bold"),
                            bg=self.colors['card'], fg=color)
            label.pack(pady=5)
            self.metric_labels[key] = label

        # Configure grid
        for i in range(4):
            metrics_grid.grid_columnconfigure(i, weight=1)
        for i in range(2):
            metrics_grid.grid_rowconfigure(i, weight=1)

    def setup_preview_tab(self):
        """Setup data preview with clean and original data"""
        # Control frame
        control_frame = tk.Frame(self.preview_tab, bg=self.colors['secondary'])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(control_frame, text="Show:", bg=self.colors['secondary'],
                fg=self.colors['text']).pack(side=tk.LEFT, padx=5)

        self.preview_type = ttk.Combobox(control_frame, values=['Original Data', 'Cleaned Data'], width=15)
        self.preview_type.pack(side=tk.LEFT, padx=5)
        self.preview_type.set('Original Data')
        self.preview_type.bind('<<ComboboxSelected>>', lambda e: self.refresh_preview())

        tk.Label(control_frame, text="Rows:", bg=self.colors['secondary'],
                fg=self.colors['text']).pack(side=tk.LEFT, padx=5)

        self.preview_rows = tk.Spinbox(control_frame, from_=10, to=500, width=8)
        self.preview_rows.pack(side=tk.LEFT, padx=5)
        self.preview_rows.delete(0, tk.END)
        self.preview_rows.insert(0, "50")

        tk.Button(control_frame, text="🔄 Refresh", command=self.refresh_preview,
                 bg=self.colors['accent'], fg='white').pack(side=tk.LEFT, padx=10)

        # Treeview frame
        self.preview_frame = tk.Frame(self.preview_tab, bg=self.colors['secondary'])
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.preview_tree = ttk.Treeview(self.preview_frame)
        vsb = ttk.Scrollbar(self.preview_frame, orient="vertical", command=self.preview_tree.yview)
        hsb = ttk.Scrollbar(self.preview_frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.preview_tree.pack(side="left", fill="both", expand=True)

    def setup_clean_tab(self):
        """Setup cleaning results comparison"""
        compare_frame = tk.Frame(self.clean_tab, bg=self.colors['secondary'])
        compare_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Before frame
        before_frame = tk.LabelFrame(compare_frame, text="📋 BEFORE CLEANING",
                                     font=("Arial", 12, "bold"), bg=self.colors['card'],
                                     fg=self.colors['warning'], padx=10, pady=10)
        before_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.before_text = scrolledtext.ScrolledText(before_frame, wrap=tk.WORD,
                                                      bg=self.colors['card_dark'],
                                                      fg=self.colors['text_light'],
                                                      font=("Consolas", 9))
        self.before_text.pack(fill=tk.BOTH, expand=True)

        # After frame
        after_frame = tk.LabelFrame(compare_frame, text="✨ AFTER CLEANING",
                                    font=("Arial", 12, "bold"), bg=self.colors['card'],
                                    fg=self.colors['success'], padx=10, pady=10)
        after_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.after_text = scrolledtext.ScrolledText(after_frame, wrap=tk.WORD,
                                                     bg=self.colors['card_dark'],
                                                     fg=self.colors['text_light'],
                                                     font=("Consolas", 9))
        self.after_text.pack(fill=tk.BOTH, expand=True)

    def setup_viz_tab(self):
        """Setup visualization with multiple window option"""
        viz_frame = tk.Frame(self.viz_tab, bg=self.colors['secondary'])
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(viz_frame, text="📊 DATA VISUALIZATION",
                font=("Arial", 16, "bold"), bg=self.colors['secondary'],
                fg=self.colors['accent']).pack()

        tk.Label(viz_frame, text="Click any button to open visualization in a new window",
                font=("Arial", 10), bg=self.colors['secondary'], fg=self.colors['text_light']).pack(pady=5)

        # Buttons frame
        btn_frame = tk.Frame(viz_frame, bg=self.colors['secondary'])
        btn_frame.pack(pady=30)

        buttons = [
            ("📊 Quality Dashboard", "#e94560", self.open_dashboard_window),
            ("🔥 Missing Values Heatmap", "#ff6b6b", self.open_heatmap_window),
            ("📈 Correlation Matrix", "#4ecdc4", self.open_correlation_window),
            ("📉 Data Distribution", "#f9ca24", self.open_distribution_window),
            ("🥧 Data Types Pie Chart", "#9b59b6", self.open_piechart_window),
            ("📊 Box Plots", "#3498db", self.open_boxplot_window)
        ]

        for text, color, command in buttons:
            btn = tk.Button(btn_frame, text=text, command=command,
                           bg=color, fg='white', font=("Arial", 11, "bold"),
                           width=25, height=2)
            btn.pack(pady=10)

    def setup_anomaly_tab(self):
        """Setup anomaly detection tab"""
        self.anomaly_text = scrolledtext.ScrolledText(self.anomaly_tab, wrap=tk.WORD,
                                                       bg=self.colors['card_dark'],
                                                       fg=self.colors['text_light'],
                                                       font=("Consolas", 10))
        self.anomaly_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.anomaly_text.insert(tk.END, "🤖 AI-Powered Anomaly Detection\n\nClick 'Detect Anomalies' in the left panel to start.\n")

    def setup_logs_tab(self):
        """Setup logs tab"""
        self.log_text = scrolledtext.ScrolledText(self.logs_tab, wrap=tk.WORD,
                                                   bg=self.colors['card_dark'],
                                                   fg=self.colors['text_light'],
                                                   font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def add_log(self, message):
        """Add log message"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def update_status(self, message, progress=False):
        """Update status bar"""
        self.status_label.config(text=f"🔄 {message}" if progress else f"✅ {message}")
        if progress:
            self.progress.start()
        else:
            self.progress.stop()
        self.root.update_idletasks()

    def upload_file(self):
        """Upload file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")]
        )
        if not file_path:
            return

        self.update_status(f"Loading {os.path.basename(file_path)}...", True)

        def load():
            try:
                self.current_file_path = file_path
                file_type = 'csv' if file_path.endswith('.csv') else 'excel'
                self.data_cleaner = EnhancedDataCleaner(log_callback=self.add_log)
                self.current_dataset = self.data_cleaner.load_dataset(file_path, file_type)
                self.original_dataset = self.current_dataset.copy()
                self.root.after(0, self.on_load_complete)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.update_status("Ready", False))

        Thread(target=load).start()

    def on_load_complete(self):
        """Post load operations"""
        self.update_status(f"Loaded {self.current_dataset.shape[0]:,} rows, {self.current_dataset.shape[1]} columns", False)

        self.file_info.delete(1.0, tk.END)
        self.file_info.insert(1.0,
            f"📄 File: {os.path.basename(self.current_file_path)}\n"
            f"📊 Rows: {self.current_dataset.shape[0]:,}\n"
            f"📋 Columns: {self.current_dataset.shape[1]}\n"
            f"💾 Size: {self.current_dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            f"⚠️ Missing: {self.current_dataset.isnull().sum().sum():,}\n"
            f"🔄 Duplicates: {self.current_dataset.duplicated().sum():,}"
        )

        self.update_dashboard()
        self.refresh_preview()
        messagebox.showinfo("Success", f"Dataset loaded!\nShape: {self.current_dataset.shape}")

    def update_dashboard(self):
        """Update dashboard metrics"""
        if self.current_dataset is None:
            return

        quality = self.quality_scorer.calculate_quality_score(self.current_dataset)

        self.score_label.config(text=f"{quality['total_score']}%")
        self.grade_label.config(text=f"Grade: {quality['grade']} - {quality['description']}")

        self.metric_labels['rows_label'].config(text=f"{len(self.current_dataset):,}")
        self.metric_labels['cols_label'].config(text=f"{len(self.current_dataset.columns)}")
        self.metric_labels['missing_label'].config(text=f"{self.current_dataset.isnull().sum().sum():,}")
        self.metric_labels['duplicate_label'].config(text=f"{self.current_dataset.duplicated().sum():,}")
        self.metric_labels['complete_label'].config(text=f"{quality['metrics']['completeness']:.1f}%")
        self.metric_labels['unique_label'].config(text=f"{quality['metrics']['uniqueness']:.1f}%")
        self.metric_labels['memory_label'].config(text=f"{self.current_dataset.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        self.metric_labels['dtypes_label'].config(text=f"{len(self.current_dataset.dtypes.unique())}")

    def refresh_preview(self):
        """Refresh preview table"""
        if self.current_dataset is None:
            return

        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        # Show either original or cleaned data
        if self.preview_type.get() == 'Cleaned Data' and self.cleaned_dataset is not None:
            data = self.cleaned_dataset
        else:
            data = self.current_dataset

        n_rows = int(self.preview_rows.get())
        columns = list(data.columns)

        self.preview_tree["columns"] = columns
        self.preview_tree["show"] = "headings"

        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=120, minwidth=80)

        for idx, row in data.head(n_rows).iterrows():
            values = [str(v)[:50] if not pd.isna(v) else "NULL" for v in row]
            self.preview_tree.insert("", "end", values=values)

    def start_cleaning(self):
        """Start cleaning"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Please upload a dataset first")
            return

        self.update_status("Cleaning in progress...", True)

        def clean():
            try:
                config = {
                    'standardize_columns': self.clean_vars["✓ Standardize Column Names"].get(),
                    'remove_duplicates': self.clean_vars["✓ Remove Duplicate Rows"].get(),
                    'handle_missing': self.clean_vars["✓ Handle Missing Values"].get(),
                    'correct_types': self.clean_vars["✓ Correct Data Types"].get(),
                    'clean_text': self.clean_vars["✓ Clean Text Data"].get(),
                    'missing_strategy': self.missing_strategy.get(),
                    'missing_threshold': 50
                }

                self.cleaned_dataset = self.data_cleaner.clean_dataset(self.current_dataset.copy(), config)
                self.cleaning_summary = self.data_cleaner.cleaning_summary
                self.root.after(0, self.on_clean_complete)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.update_status("Ready", False))

        Thread(target=clean).start()

    def on_clean_complete(self):
        """Post cleaning"""
        self.update_status("Cleaning completed!", False)

        # Show comparison
        self.before_text.delete(1.0, tk.END)
        self.after_text.delete(1.0, tk.END)

        self.before_text.insert(tk.END, f"Original Dataset:\n{'='*50}\n")
        self.before_text.insert(tk.END, f"Rows: {self.original_dataset.shape[0]:,}\n")
        self.before_text.insert(tk.END, f"Columns: {self.original_dataset.shape[1]}\n")
        self.before_text.insert(tk.END, f"Missing: {self.original_dataset.isnull().sum().sum():,}\n")
        self.before_text.insert(tk.END, f"Duplicates: {self.original_dataset.duplicated().sum():,}\n\n")

        self.after_text.insert(tk.END, f"Cleaned Dataset:\n{'='*50}\n")
        self.after_text.insert(tk.END, f"Rows: {self.cleaned_dataset.shape[0]:,}\n")
        self.after_text.insert(tk.END, f"Columns: {self.cleaned_dataset.shape[1]}\n")
        self.after_text.insert(tk.END, f"Missing: {self.cleaned_dataset.isnull().sum().sum():,}\n")
        self.after_text.insert(tk.END, f"Duplicates: {self.cleaned_dataset.duplicated().sum():,}\n\n")

        self.after_text.insert(tk.END, f"Operations Performed:\n{'='*50}\n")
        for op in self.cleaning_summary.get('cleaning_operations', []):
            self.after_text.insert(tk.END, f"✓ {op}\n")

        # Ask to replace
        if messagebox.askyesno("Cleaning Complete",
            f"Before: {self.cleaning_summary['original_shape'][0]:,} rows\n"
            f"After: {self.cleaning_summary['cleaned_shape'][0]:,} rows\n\n"
            f"Replace original with cleaned version?"):
            self.current_dataset = self.cleaned_dataset
            self.update_dashboard()
            self.refresh_preview()
            messagebox.showinfo("Success", "Dataset updated!")

        self.notebook.select(self.clean_tab)

    def save_cleaned_data(self):
        """Save cleaned data"""
        data = self.cleaned_dataset if self.cleaned_dataset is not None else self.current_dataset
        if data is None:
            messagebox.showwarning("No Data", "No data to save")
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])

        if path:
            try:
                if path.endswith('.csv'):
                    data.to_csv(path, index=False)
                else:
                    data.to_excel(path, index=False)
                messagebox.showinfo("Success", f"Saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def detect_anomalies(self):
        """Run anomaly detection"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        self.update_status("Detecting anomalies...", True)

        def detect():
            try:
                self.anomaly_text.delete(1.0, tk.END)
                results = self.anomaly_detector.detect_anomalies(self.current_dataset)

                output = f"""
{'='*60}
🤖 AI-POWERED ANOMALY DETECTION RESULTS
{'='*60}

📊 DATASET OVERVIEW:
   • Total Rows: {len(self.current_dataset):,}
   • Total Columns: {len(self.current_dataset.columns)}
   • Numeric Columns: {len(results['affected_columns'])}

🔍 ANOMALY DETECTION:
   • Anomaly Count: {results['anomaly_count']} rows
   • Anomaly Percentage: {results['anomaly_percentage']:.2f}%

📈 AFFECTED COLUMNS:
"""
                for col in results['affected_columns'][:10]:
                    output += f"   • {col}\n"

                if results['anomaly_count'] > 0:
                    output += f"""
⚠️  RECOMMENDATIONS:
   • Review {results['anomaly_count']} anomalous rows
   • Consider removing or investigating anomalies
   • Run cleaning operations to fix issues
"""
                else:
                    output += f"""
✅ EXCELLENT! No anomalies detected!
   Your data is clean and well-structured.
"""

                self.root.after(0, lambda: self.anomaly_text.insert(tk.END, output))
                self.root.after(0, lambda: self.update_status("Ready", False))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.update_status("Ready", False))

        Thread(target=detect).start()

    def open_dashboard_window(self):
        """Open dashboard in new window"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        win = tk.Toplevel(self.root)
        win.title("Data Quality Dashboard")
        win.geometry("900x600")
        win.configure(bg=self.colors['secondary'])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Data Quality Dashboard', fontsize=14, fontweight='bold')

        quality = self.quality_scorer.calculate_quality_score(self.current_dataset)

        # Quality gauge
        axes[0, 0].pie([quality['total_score'], 100-quality['total_score']],
                       labels=[f"Score: {quality['total_score']}%", ""],
                       colors=['#4ecdc4', '#ff6b6b'], autopct='%1.1f%%')
        axes[0, 0].set_title('Overall Quality Score')

        # Metrics bar chart
        metrics = quality['metrics']
        axes[0, 1].bar(metrics.keys(), metrics.values(), color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].set_title('Quality Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Missing values
        missing_data = self.current_dataset.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            axes[1, 0].barh(missing_data.index[:10], missing_data.values[:10], color='#ff6b6b')
            axes[1, 0].set_title('Missing Values by Column')

        # Data types
        dtype_counts = self.current_dataset.dtypes.value_counts()
        axes[1, 1].pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index], autopct='%1.1f%%')
        axes[1, 1].set_title('Data Type Distribution')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def open_heatmap_window(self):
        """Open missing values heatmap"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        win = tk.Toplevel(self.root)
        win.title("Missing Values Heatmap")
        win.geometry("800x600")
        win.configure(bg=self.colors['secondary'])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.current_dataset.isnull(), yticklabels=False, cbar=True, ax=ax, cmap='viridis')
        ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def open_correlation_window(self):
        """Open correlation matrix"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        numeric_cols = self.current_dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            messagebox.showwarning("Insufficient Data", "Need at least 2 numeric columns")
            return

        win = tk.Toplevel(self.root)
        win.title("Correlation Matrix")
        win.geometry("800x700")
        win.configure(bg=self.colors['secondary'])

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self.current_dataset[numeric_cols].corr()
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def open_distribution_window(self):
        """Open distribution plots"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        numeric_cols = self.current_dataset.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            messagebox.showwarning("No Data", "No numeric columns found")
            return

        win = tk.Toplevel(self.root)
        win.title("Data Distribution")
        win.geometry("1000x800")
        win.configure(bg=self.colors['secondary'])

        n_cols = min(len(numeric_cols), 6)
        fig, axes = plt.subplots((n_cols+1)//2, 2, figsize=(12, 4*((n_cols+1)//2)))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            axes[i].hist(self.current_dataset[col].dropna(), bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')

        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def open_piechart_window(self):
        """Open pie chart for data types"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        win = tk.Toplevel(self.root)
        win.title("Data Type Distribution")
        win.geometry("600x500")
        win.configure(bg=self.colors['secondary'])

        fig, ax = plt.subplots(figsize=(8, 6))
        dtype_counts = self.current_dataset.dtypes.value_counts()
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        ax.pie(dtype_counts.values, labels=[str(d) for d in dtype_counts.index],
               autopct='%1.1f%%', colors=colors[:len(dtype_counts)])
        ax.set_title('Data Type Distribution', fontsize=14, fontweight='bold')

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def open_boxplot_window(self):
        """Open box plots"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "Upload dataset first")
            return

        numeric_cols = self.current_dataset.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) == 0:
            messagebox.showwarning("No Data", "No numeric columns found")
            return

        win = tk.Toplevel(self.root)
        win.title("Box Plots")
        win.geometry("1000x600")
        win.configure(bg=self.colors['secondary'])

        fig, ax = plt.subplots(figsize=(12, 6))
        data_to_plot = [self.current_dataset[col].dropna() for col in numeric_cols]
        bp = ax.boxplot(data_to_plot, labels=numeric_cols, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)

        ax.set_title('Box Plots for Numeric Columns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Values')
        ax.tick_params(axis='x', rotation=45)

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

    def export_report(self):
        """Export report"""
        if self.current_dataset is None:
            messagebox.showwarning("No Data", "No data to export")
            return

        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            try:
                quality = self.quality_scorer.calculate_quality_score(self.current_dataset)

                with open(path, 'w', encoding='utf-8') as f:
                    f.write("AUTO DATA CLEANER PRO - QUALITY REPORT\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"File: {os.path.basename(self.current_file_path) if self.current_file_path else 'N/A'}\n\n")

                    f.write("DATASET SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Rows: {len(self.current_dataset):,}\n")
                    f.write(f"Columns: {len(self.current_dataset.columns)}\n")
                    f.write(f"Missing Values: {self.current_dataset.isnull().sum().sum():,}\n")
                    f.write(f"Duplicate Rows: {self.current_dataset.duplicated().sum():,}\n\n")

                    f.write("QUALITY SCORE\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Overall: {quality['total_score']}%\n")
                    f.write(f"Grade: {quality['grade']}\n")
                    f.write(f"Description: {quality['description']}\n\n")

                    f.write("METRICS\n")
                    f.write("-" * 40 + "\n")
                    for metric, score in quality['metrics'].items():
                        f.write(f"{metric.title()}: {score}%\n")

                messagebox.showinfo("Success", f"Report saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def run(self):
        """Run application"""
        self.root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("AUTO DATA CLEANER PRO - ENTERPRISE EDITION")
    print("=" * 60)
    print("Starting application...")

    try:
        app = ProfessionalDataCleaner()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")