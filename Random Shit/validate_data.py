"""
DATA VALIDATION SCRIPT
======================
Validates downloaded BigQuery power data for quality and completeness.
"""

import pandas as pd
import os
from datetime import datetime

def validate_downloaded_data(filepath):
    """
    Comprehensive validation of downloaded power data.
    
    Args:
        filepath: Path to the CSV file to validate
    
    Returns:
        dict: Validation results
    """
    
    print("\n" + "="*70)
    print("DATA VALIDATION REPORT")
    print("="*70)
    print(f"\nFile: {os.path.basename(filepath)}")
    
    results = {
        'passed': [],
        'warnings': [],
        'failed': []
    }
    
    # Check 1: File exists
    print("\n[1/10] Checking file existence...")
    if os.path.exists(filepath):
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"        ✅ File exists ({file_size_mb:.1f} MB)")
        results['passed'].append(f"File exists: {file_size_mb:.1f} MB")
        
        if file_size_mb < 10:
            results['warnings'].append(f"File size small ({file_size_mb:.1f} MB) - may be incomplete")
        elif file_size_mb > 200:
            results['warnings'].append(f"File size large ({file_size_mb:.1f} MB) - check for duplicates")
    else:
        print(f"        ❌ File not found")
        results['failed'].append("File not found")
        return results
    
    # Load data
    print("\n[2/10] Loading data...")
    try:
        df = pd.read_csv(filepath)
        print(f"        ✅ Data loaded ({len(df):,} records)")
        results['passed'].append(f"Data loaded: {len(df):,} records")
    except Exception as e:
        print(f"        ❌ Failed to load: {e}")
        results['failed'].append(f"Failed to load: {e}")
        return results
    
    # Check 3: Record count
    print("\n[3/10] Checking record count...")
    expected_min = 80000  # At least 80k for May 2019 hourly data
    expected_max = 1000000  # Less than 1M
    
    if expected_min <= len(df) <= expected_max:
        print(f"        ✅ Record count within expected range")
        results['passed'].append(f"Record count: {len(df):,}")
    elif len(df) < expected_min:
        print(f"        ⚠️  Record count low ({len(df):,} < {expected_min:,})")
        results['warnings'].append(f"Low record count: {len(df):,}")
    else:
        print(f"        ⚠️  Record count high ({len(df):,} > {expected_max:,})")
        results['warnings'].append(f"High record count: {len(df):,}")
    
    # Check 4: Required columns
    print("\n[4/10] Checking required columns...")
    required_cols = ['time', 'measured_power_util', 'cell', 'pdu']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if not missing_cols:
        print(f"        ✅ All required columns present")
        results['passed'].append("All required columns present")
    else:
        print(f"        ❌ Missing columns: {missing_cols}")
        results['failed'].append(f"Missing columns: {missing_cols}")
    
    print(f"        Available columns: {list(df.columns)}")
    
    # Check 5: Date range
    print("\n[5/10] Checking date range...")
    if 'time' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['time'])
            min_date = df['timestamp'].min()
            max_date = df['timestamp'].max()
            days_span = (max_date - min_date).days
            
            print(f"        Start: {min_date}")
            print(f"        End: {max_date}")
            print(f"        Span: {days_span} days")
            
            if min_date.year == 2019 and min_date.month == 5:
                print(f"        ✅ Date range correct (May 2019)")
                results['passed'].append(f"Date range: May 2019 ({days_span} days)")
            else:
                print(f"        ⚠️  Unexpected date range")
                results['warnings'].append(f"Date range: {min_date} to {max_date}")
                
        except Exception as e:
            print(f"        ⚠️  Could not parse dates: {e}")
            results['warnings'].append("Date parsing failed")
    
    # Check 6: Cell coverage
    print("\n[6/10] Checking cell coverage...")
    if 'cell' in df.columns:
        cells = df['cell'].unique()
        expected_cells = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        missing_cells = [c for c in expected_cells if c not in cells]
        
        print(f"        Cells found: {sorted(cells)}")
        print(f"        Count: {len(cells)}/8")
        
        if len(cells) >= 6:
            print(f"        ✅ Good cell coverage")
            results['passed'].append(f"Cells: {len(cells)}/8")
        else:
            print(f"        ⚠️  Limited cell coverage")
            results['warnings'].append(f"Only {len(cells)}/8 cells")
    
    # Check 7: PDU coverage
    print("\n[7/10] Checking PDU coverage...")
    if 'pdu' in df.columns:
        pdus = df['pdu'].unique()
        print(f"        PDUs found: {len(pdus)}")
        
        if len(pdus) >= 50:
            print(f"        ✅ Good PDU coverage")
            results['passed'].append(f"PDUs: {len(pdus)}")
        else:
            print(f"        ⚠️  Limited PDU coverage")
            results['warnings'].append(f"Only {len(pdus)} PDUs")
    
    # Check 8: Power utilization values
    print("\n[8/10] Checking power utilization values...")
    if 'measured_power_util' in df.columns:
        power_data = df['measured_power_util'].dropna()
        
        if len(power_data) > 0:
            min_power = power_data.min()
            max_power = power_data.max()
            mean_power = power_data.mean()
            null_count = df['measured_power_util'].isna().sum()
            
            print(f"        Range: {min_power:.4f} to {max_power:.4f}")
            print(f"        Mean: {mean_power:.4f}")
            print(f"        Null values: {null_count:,} ({null_count/len(df)*100:.1f}%)")
            
            # Validate range
            if 0 <= min_power <= 1 and 0 <= max_power <= 1:
                print(f"        ✅ Values within valid range [0, 1]")
                results['passed'].append("Power values in valid range")
            else:
                print(f"        ❌ Values outside expected range")
                results['failed'].append(f"Power values invalid: {min_power} to {max_power}")
            
            # Check for too many nulls
            if null_count / len(df) > 0.1:
                results['warnings'].append(f"High null percentage: {null_count/len(df)*100:.1f}%")
        else:
            print(f"        ❌ No valid power data")
            results['failed'].append("No valid power data")
    
    # Check 9: Duplicate timestamps
    print("\n[9/10] Checking for duplicates...")
    if 'time' in df.columns and 'cell' in df.columns and 'pdu' in df.columns:
        duplicates = df.duplicated(subset=['time', 'cell', 'pdu']).sum()
        
        if duplicates == 0:
            print(f"        ✅ No duplicate records")
            results['passed'].append("No duplicates")
        else:
            print(f"        ⚠️  Found {duplicates:,} duplicate records")
            results['warnings'].append(f"Duplicates: {duplicates:,}")
    
    # Check 10: Data completeness
    print("\n[10/10] Checking data completeness...")
    total_nulls = df.isnull().sum().sum()
    total_cells = len(df) * len(df.columns)
    completeness = (1 - total_nulls / total_cells) * 100
    
    print(f"        Completeness: {completeness:.1f}%")
    
    if completeness >= 95:
        print(f"        ✅ Excellent data completeness")
        results['passed'].append(f"Completeness: {completeness:.1f}%")
    elif completeness >= 85:
        print(f"        ⚠️  Acceptable completeness")
        results['warnings'].append(f"Completeness: {completeness:.1f}%")
    else:
        print(f"        ❌ Poor data completeness")
        results['failed'].append(f"Low completeness: {completeness:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n✅ PASSED ({len(results['passed'])}):")
    for item in results['passed']:
        print(f"   - {item}")
    
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for item in results['warnings']:
            print(f"   - {item}")
    
    if results['failed']:
        print(f"\n❌ FAILED ({len(results['failed'])}):")
        for item in results['failed']:
            print(f"   - {item}")
    
    # Overall verdict
    print("\n" + "="*70)
    if not results['failed'] and len(results['warnings']) <= 2:
        print("OVERALL: ✅ DATA QUALITY EXCELLENT - Ready for modeling")
    elif not results['failed']:
        print("OVERALL: ⚠️  DATA QUALITY ACCEPTABLE - Review warnings")
    else:
        print("OVERALL: ❌ DATA QUALITY ISSUES - Address failures before proceeding")
    print("="*70 + "\n")
    
    return results


def main():
    """Main validation routine."""
    
    # Check for power data files
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data_Sources')
    
    # Look for BigQuery power data files
    possible_files = [
        'google_power_2019_full.csv',
        'google_power_2019_sample.csv',
        'google_power_2019.csv',
        'bigquery_power_2019.csv'
    ]
    
    found_files = []
    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            found_files.append(filepath)
    
    if not found_files:
        print("\n" + "="*70)
        print("⚠️  NO POWER DATA FILES FOUND")
        print("="*70)
        print("\nSearched for:")
        for filename in possible_files:
            print(f"  - {filename}")
        print("\nRun the download script first:")
        print("  .venv\\Scripts\\python.exe Model_Files\\download_bigquery_data.py")
        print("\n")
        return
    
    # Validate each file found
    for filepath in found_files:
        validate_downloaded_data(filepath)
        
        if len(found_files) > 1:
            print("\n" + "-"*70 + "\n")


if __name__ == "__main__":
    main()
