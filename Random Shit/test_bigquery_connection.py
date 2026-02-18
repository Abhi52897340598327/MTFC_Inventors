"""
BIGQUERY CONNECTION TEST
========================
Tests if your local environment can connect to BigQuery.
Run this AFTER completing authentication (gcloud auth application-default login).
"""

from google.cloud import bigquery
import sys

def test_connection(project_id):
    """Test BigQuery connection with a simple query."""
    
    print("\n" + "="*70)
    print("TESTING BIGQUERY CONNECTION")
    print("="*70)
    print(f"\nProject ID: {project_id}")
    
    try:
        # Create client (uses Application Default Credentials)
        print("\n[1/3] Creating BigQuery client...")
        client = bigquery.Client(project=project_id)
        print("      ✅ Client created successfully")
        
        # Test query - count rows in one table
        print("\n[2/3] Testing query execution...")
        # Note: Google's public dataset is in EU location
        query = """
        SELECT COUNT(*) as total_rows
        FROM `google.com:google-cluster-data.powerdata_2019.cella_pdu01`
        LIMIT 1
        """
        
        # Specify location for public dataset
        job_config = bigquery.QueryJobConfig(use_legacy_sql=False)
        result = client.query(query, job_config=job_config, location='EU').result()
        print("      ✅ Query executed successfully")
        
        # Get results
        print("\n[3/3] Retrieving results...")
        for row in result:
            print(f"      ✅ Data accessible! Table has {row.total_rows:,} rows")
        
        print("\n" + "="*70)
        print("SUCCESS! BigQuery connection is working perfectly! ✅")
        print("="*70)
        print("\nYou're ready to run the full download script:")
        print("  .venv\\Scripts\\python.exe Model_Files\\download_bigquery_data.py")
        print("\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ CONNECTION FAILED")
        print("="*70)
        print(f"\nError: {str(e)}\n")
        
        # Provide helpful troubleshooting
        error_str = str(e).lower()
        
        if "could not automatically determine credentials" in error_str or "default credentials" in error_str:
            print("SOLUTION: Authentication required")
            print("Run this command in PowerShell:")
            print("  gcloud auth application-default login")
            print("\nThen try this test script again.\n")
            
        elif "project" in error_str or "project id" in error_str:
            print("SOLUTION: Invalid project ID")
            print("Check your project ID:")
            print("  gcloud config list")
            print("\nUpdate PROJECT_ID in this script with the correct ID.\n")
            
        elif "permission" in error_str or "access" in error_str:
            print("SOLUTION: BigQuery API not enabled")
            print("Enable it here:")
            print("  https://console.cloud.google.com/apis/library/bigquery.googleapis.com")
            print("\nMake sure you're using the correct project.\n")
            
        else:
            print("SOLUTION: Check the error message above")
            print("Common issues:")
            print("  1. Internet connection")
            print("  2. Firewall blocking Google Cloud services")
            print("  3. Incorrect project ID")
            print("\n")
        
        return False


if __name__ == "__main__":
    # PROJECT ID - Update this with your actual project ID
    PROJECT_ID = "datacenter-forecasting"
    
    if PROJECT_ID == "YOUR_PROJECT_ID_HERE":
        print("\n" + "="*70)
        print("⚠️  CONFIGURATION REQUIRED")
        print("="*70)
        print("\nPlease update this script:")
        print("  Line 89: PROJECT_ID = 'YOUR_PROJECT_ID_HERE'")
        print("\nReplace with your actual project ID from Google Cloud Console.")
        print("\nTo find your project ID:")
        print("  1. Run: gcloud config list")
        print("  2. Look for 'project = YOUR-PROJECT-ID'")
        print("  3. Copy that ID into this script")
        print("\n")
        sys.exit(1)
    
    # Run test
    success = test_connection(PROJECT_ID)
    sys.exit(0 if success else 1)
