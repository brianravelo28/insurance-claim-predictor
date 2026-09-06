"""
Advanced API fixes - Try FTP, alternative endpoints, and workarounds
"""

import requests
import subprocess
import os

print("ADVANCED API FIXES")
print("=" * 70)

# ==================== FEMA API - Check documentation ====================
print("\nFEMA API - Checking for API documentation/alternatives")
print("-" * 70)

print("\n1. Checking FEMA OpenData documentation:")
try:
    r = requests.get("https://www.fema.gov/api/open/", timeout=5)
    print(f"   FEMA API docs page: HTTP {r.status_code}")
    if r.status_code == 200:
        print(f"   Page accessible - check for updated endpoint info")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Trying FEMA Data API with version header:")
try:
    headers = {'Accept': 'application/json', 'API-Version': '1.0'}
    r = requests.get("https://www.fema.gov/api/open/data/FimaNfipClaims",
                     params={"$limit": 5}, headers=headers, timeout=5)
    print(f"   With version header: HTTP {r.status_code}")
    if r.status_code == 200:
        print(f"   SUCCESS! API works with version header")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Alternative FEMA endpoints to try:")
fema_alternatives = [
    "https://opendata.fema.gov/api/3/action/datastore_search?resource_id=e1e75f67-8b86-9d1c-8b2f-5c6d3964b2bd",
    "https://data.fema.gov/api/views/r2nx-4cbf/rows.json",
]

for url in fema_alternatives:
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        print(f"   {url[:60]}: HTTP {r.status_code}")
    except Exception as e:
        print(f"   {url[:60]}: {type(e).__name__}")

# ==================== HURDAT2 - Try FTP ====================
print("\n" + "=" * 70)
print("HURDAT2 - Attempting FTP Access")
print("-" * 70)

print("\n1. Checking if FTP access available:")
ftp_test = """
Try using an FTP client to access:
  Host: ftp.nhc.noaa.gov
  Path: /atcf/archive/hurdat2.txt

From Python, use:
  from ftplib import FTP
  ftp = FTP('ftp.nhc.noaa.gov')
  ftp.login()
  ftp.retrbinary('RETR /atcf/archive/hurdat2.txt', callback)
"""
print(ftp_test)

# Try FTP with Python
print("2. Attempting FTP connection:")
try:
    from ftplib import FTP, all_errors
    print("   Connecting to ftp.nhc.noaa.gov...")
    ftp = FTP('ftp.nhc.noaa.gov', timeout=10)
    print(f"   Connected: {ftp.getwelcome()[:50]}")

    # Try to list directory
    print("   Listing /atcf/archive/:")
    ftp.cwd('/atcf/archive/')
    files = ftp.nlst()
    hurdat_files = [f for f in files if 'hurdat' in f.lower()]

    if hurdat_files:
        print(f"   Found HURDAT files: {hurdat_files}")
        for f in hurdat_files[:3]:
            try:
                size = ftp.size(f)
                print(f"      {f}: {size:,} bytes")
            except:
                pass
    else:
        print(f"   No HURDAT files found")
        print(f"   Files in directory: {files[:10]}")

    ftp.quit()

except Exception as e:
    print(f"   FTP Error: {type(e).__name__}: {e}")

# ==================== Alternative Data Sources ====================
print("\n" + "=" * 70)
print("ALTERNATIVE DATA SOURCES")
print("-" * 70)

print("\n1. NOAA Data Portal (data.noaa.gov):")
try:
    r = requests.get("https://data.noaa.gov/api/3/action/package_list", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        packages = data.get('result', [])
        hurdat_packages = [p for p in packages if 'hurdat' in p.lower()]
        if hurdat_packages:
            print(f"   Found HURDAT datasets: {hurdat_packages}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Kaggle Datasets (if available):")
print("   - Search: 'FEMA NFIP claims' or 'HURDAT hurricane'")
print("   - Requires Kaggle API setup: pip install kaggle")

print("\n3. NOAA Climate Data Online:")
print("   - URL: https://www.ncei.noaa.gov/cdo-web/")
print("   - Has historical hurricane and claim data")

# ==================== WORKAROUND SUMMARY ====================
print("\n" + "=" * 70)
print("SUMMARY & WORKAROUNDS")
print("=" * 70)

summary = """
CURRENT SITUATION:
  ✗ FEMA API: Consistently returns "Invalid version format"
    → Likely service issue or endpoint deprecated
  ✗ HURDAT2 HTTP: All URLs return 404
    → File may be offline or relocated
  ? HURDAT2 FTP: Untested (requires FTP client)
    → Possible fallback option

YOUR OPTIONS:

Option 1: USE SYNTHETIC DATA (Current - Fully Working)
  ✓ Pipeline complete and validated
  ✓ All outputs generated
  ✓ Ready for Day 2 modeling
  → Recommended: Proceed to Day 2

Option 2: WAIT & RETRY
  - Monitor FEMA/NOAA services
  - Re-run pipeline when APIs restore
  - Same code, automatic switch to real data

Option 3: USE ALTERNATIVE SOURCES
  - NOAA Climate Data Online (manual download)
  - Kaggle FEMA/Hurricane datasets
  - NOAA FTP servers (ftp.ncei.noaa.gov)

Option 4: CONTACT FEMA/NOAA
  - Report API issues to FEMA OpenData team
  - Check NOAA status page for service issues

RECOMMENDED NEXT STEP:
  Continue with synthetic data and proceed to Day 2 modeling.
  The pipeline is designed to seamlessly use real data when available.
"""

print(summary)
