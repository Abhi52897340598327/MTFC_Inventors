"""
COMPREHENSIVE DATA CLEANING SCRIPT
For MTFC Virginia Datacenter Forecasting Project

This script:
1. Checks all CSV files for data quality issues
2. Applies standard data cleaning methods
3. Saves cleaned versions with '_cleaned' suffix
4. Generates a detailed data quality report
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import glob

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
CLEANED_DIR = os.path.join(DATA_DIR, 'cleaned')
os.makedirs(CLEANED_DIR, exist_ok=True)

# Data quality report storage
quality_report = []

def log_info(message):
    """Log information to console and report"""
    print(message)
    quality_report.append(message)

def standardize_column_names(df):
    """Standardize column names: lowercase, replace spaces with underscores"""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def detect_datetime_columns(df):
    """Detect and convert datetime columns"""
    datetime_keywords = ['date', 'time', 'timestamp', 'period', 'datetime']
    converted_cols = []
    
    for col in df.columns:
        if any(keyword in col.lower() for keyword in datetime_keywords):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                converted_cols.append(col)
            except:
                pass
    
    return df, converted_cols

def handle_missing_values(df, filename):
    """Identify and handle missing values"""
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    
    missing_cols = missing_summary[missing_summary > 0]
    
    if len(missing_cols) > 0:
        log_info(f"\n  Missing values found:")
        for col, count in missing_cols.items():
            pct = missing_pct[col]
            log_info(f"    - {col}: {count} ({pct}%)")
            
            # Strategy: Drop columns with >50% missing, fill numeric with median, drop rows for critical columns
            if pct > 50:
                log_info(f"      Action: Dropping column (>50% missing)")
                df = df.drop(columns=[col])
            elif df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log_info(f"      Action: Filled with median ({median_val})")
            else:
                # For categorical/text, fill with 'Unknown' or most frequent
                if df[col].dtype == 'object':
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_val)
                    log_info(f"      Action: Filled with mode ('{mode_val}')")
    else:
        log_info(f"  No missing values found")
    
    return df

def remove_duplicates(df, filename):
    """Remove duplicate rows"""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    
    if removed > 0:
        log_info(f"  Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
    else:
        log_info(f"  No duplicate rows found")
    
    return df

def detect_and_handle_outliers(df, filename):
    """Detect outliers using IQR method for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_found = False
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for more conservative outlier detection
        upper_bound = Q3 + 3 * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            if not outliers_found:
                log_info(f"\n  Outliers detected (using 3*IQR method):")
                outliers_found = True
            
            outlier_pct = (outlier_count / len(df) * 100)
            log_info(f"    - {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
            log_info(f"      Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            log_info(f"      Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # Don't remove outliers automatically - just flag them
            # For energy data, extreme values might be real (peak events, outages, etc.)
            log_info(f"      Action: Flagged but not removed (may be valid extreme events)")
    
    if not outliers_found:
        log_info(f"  No significant outliers detected")
    
    return df

def check_data_types(df, filename):
    """Check and report data types"""
    log_info(f"\n  Data types:")
    type_summary = df.dtypes.value_counts()
    for dtype, count in type_summary.items():
        log_info(f"    - {dtype}: {count} columns")
    
    return df

def clean_csv_file(filepath):
    """Clean a single CSV file"""
    filename = os.path.basename(filepath)
    log_info(f"\n{'='*70}")
    log_info(f"CLEANING: {filename}")
    log_info(f"{'='*70}")
    
    try:
        # Read file
        df = pd.read_csv(filepath, low_memory=False)
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        log_info(f"  Initial shape: {initial_rows} rows × {initial_cols} columns")
        
        # Check if file is empty
        if initial_rows == 0:
            log_info(f"  [WARNING] File is EMPTY - skipping")
            quality_report.append(f"  STATUS: EMPTY FILE - SKIPPED\n")
            return False
        
        # Step 1: Standardize column names
        df = standardize_column_names(df)
        log_info(f"  Standardized column names")
        
        # Step 2: Detect and convert datetime columns
        df, datetime_cols = detect_datetime_columns(df)
        if datetime_cols:
            log_info(f"  Converted {len(datetime_cols)} datetime columns: {datetime_cols}")
        
        # Step 3: Check data types
        df = check_data_types(df, filename)
        
        # Step 4: Handle missing values
        df = handle_missing_values(df, filename)
        
        # Step 5: Remove duplicates
        df = remove_duplicates(df, filename)
        
        # Step 6: Detect outliers (but don't remove them)
        df = detect_and_handle_outliers(df, filename)
        
        # Final statistics
        final_rows = len(df)
        final_cols = len(df.columns)
        log_info(f"\n  Final shape: {final_rows} rows × {final_cols} columns")
        log_info(f"  Rows changed: {final_rows - initial_rows} ({(final_rows - initial_rows)/initial_rows*100:.2f}%)")
        log_info(f"  Columns changed: {final_cols - initial_cols}")
        
        # Save cleaned file
        cleaned_filename = filename.replace('.csv', '_cleaned.csv')
        cleaned_filepath = os.path.join(CLEANED_DIR, cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)
        log_info(f"  [OK] Saved cleaned file: cleaned/{cleaned_filename}")
        
        quality_report.append(f"  STATUS: CLEANED SUCCESSFULLY\n")
        return True
        
    except Exception as e:
        log_info(f"  [ERROR] Failed to process: {str(e)}")
        quality_report.append(f"  STATUS: ERROR - {str(e)}\n")
        return False

def generate_summary_report():
    """Generate overall summary report"""
    log_info(f"\n{'='*70}")
    log_info(f"DATA QUALITY SUMMARY REPORT")
    log_info(f"{'='*70}")
    log_info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"Data directory: {DATA_DIR}")
    log_info(f"Cleaned files saved to: {CLEANED_DIR}")
    log_info(f"{'='*70}")
    
    # Save full report to file
    report_path = os.path.join(DATA_DIR, 'data_quality_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(quality_report))
    
    log_info(f"\nFull report saved to: data_quality_report.txt")

def main():
    """Main cleaning workflow"""
    print("\n" + "="*70)
    print("MTFC PROJECT - COMPREHENSIVE DATA CLEANING")
    print("="*70)
    print("\nThis script will:")
    print("  [1] Check all CSV files for quality issues")
    print("  [2] Standardize column names")
    print("  [3] Handle missing values")
    print("  [4] Remove duplicate rows")
    print("  [5] Detect outliers (flagged, not removed)")
    print("  [6] Save cleaned versions to cleaned/ folder")
    print("  [7] Generate data quality report")
    print("\n" + "="*70)
    
    input("\nPress Enter to start cleaning...\n")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    log_info(f"\nFound {len(csv_files)} CSV files to process\n")
    
    success_count = 0
    error_count = 0
    empty_count = 0
    
    # Process each file
    for filepath in sorted(csv_files):
        result = clean_csv_file(filepath)
        if result:
            success_count += 1
        elif result is False and "EMPTY" in quality_report[-1]:
            empty_count += 1
        else:
            error_count += 1
    
    # Generate summary
    log_info(f"\n{'='*70}")
    log_info(f"CLEANING COMPLETE!")
    log_info(f"{'='*70}")
    log_info(f"  Successfully cleaned: {success_count} files")
    log_info(f"  Empty files skipped: {empty_count} files")
    log_info(f"  Errors encountered: {error_count} files")
    log_info(f"{'='*70}")
    
    generate_summary_report()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. REVIEW THE REPORT:
   - Check data_quality_report.txt for detailed findings
   - Verify that cleaning decisions make sense for your analysis

2. USE CLEANED DATA:
   - Cleaned files are in Data_Sources/cleaned/ folder
   - All have '_cleaned.csv' suffix
   - Original files are preserved in Data_Sources/

3. KEY DATASETS FOR MTFC PROJECT:
   - pjm_hourly_demand_2019_2024_cleaned.csv (52K hours for grid stress)
   - virginia_co2_emissions_2015_2023_cleaned.csv (carbon analysis)
   - synthetic_datacenter_power_full_year_2019_cleaned.csv (8,760 hours)
   - virginia_electricity_consumption_2015_2024_cleaned.csv (baseline trends)

4. SPECIAL ATTENTION:
   - Empty files were flagged (e.g., epa_temperature_loudoun_2019.csv)
   - Outliers were DETECTED but NOT REMOVED (energy data may have real extremes)
   - Missing values handled: median for numeric, mode for categorical

5. READY FOR ANALYSIS:
   - Data is now standardized and ready for modeling
   - Start with exploratory data analysis (EDA)
   - Then proceed to forecasting and sensitivity analysis
    """)
    
    print("="*70)

if __name__ == "__main__":
    main()
