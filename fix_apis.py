"""
Fix FEMA and HURDAT2 API access
Test various approaches and document working solutions
"""

import requests
import json
from urllib.parse import urlencode

print("API FIX INVESTIGATION")
print("=" * 70)

# ==================== FEMA API FIXES ====================
print("\nFEMA API - Testing different approaches")
print("-" * 70)

fema_base = "https://www.fema.gov/api/open/data/FimaNfipClaims"

# Try 1: No filter at all
print("\n1. Base request (no parameters):")
try:
    r = requests.get(fema_base, timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   SUCCESS! Response size: {len(r.text)} bytes")
        data = r.json() if r.headers.get('content-type', '').find('json') > -1 else []
        if isinstance(data, list):
            print(f"   Returned {len(data)} records")
        elif isinstance(data, dict) and 'features' in data:
            print(f"   Returned {len(data['features'])} features")
    else:
        print(f"   Error: {r.text[:150]}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Try 2: Simple limit parameter
print("\n2. With $limit only:")
try:
    r = requests.get(fema_base, params={"$limit": 10}, timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   SUCCESS! Response size: {len(r.text)} bytes")
        data = r.json()
        count = len(data) if isinstance(data, list) else len(data.get('features', []))
        print(f"   Returned {count} records")
    else:
        print(f"   Error: {r.text[:150]}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Try 3: Filter with state only (no datetime)
print("\n3. Filter by state only (no datetime):")
try:
    params = {"$filter": "state eq 'FL'", "$limit": 10}
    r = requests.get(fema_base, params=params, timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   SUCCESS! Found records in Florida")
        data = r.json()
        count = len(data) if isinstance(data, list) else len(data.get('features', []))
        print(f"   Returned {count} records")
    else:
        print(f"   Error: {r.text[:150]}")
except Exception as e:
    print(f"   Error: {type(e).__name__}: {e}")

# Try 4: Different datetime formats
datetime_formats = [
    ("2000-01-01", "dateOfLoss gt 2000-01-01"),
    ("2000/01/01", "dateOfLoss gt 2000/01/01"),
    ("datetime'2000-01-01'", "dateOfLoss gt datetime'2000-01-01'"),
]

print("\n4. Testing different datetime formats:")
for fmt, filter_str in datetime_formats:
    try:
        params = {"$filter": f"state eq 'FL' and {filter_str}", "$limit": 5}
        r = requests.get(fema_base, params=params, timeout=5)
        print(f"   Format '{fmt}': HTTP {r.status_code}", end="")
        if r.status_code == 200:
            data = r.json()
            count = len(data) if isinstance(data, list) else len(data.get('features', []))
            print(f" - {count} records")
        else:
            error = r.json().get('error', 'Unknown') if r.headers.get('content-type', '').find('json') > -1 else r.text[:50]
            print(f" - Error: {error}")
    except Exception as e:
        print(f"   Format '{fmt}': {type(e).__name__}")

# Try 5: Check if API needs authentication or User-Agent
print("\n5. With custom User-Agent:")
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    params = {"$filter": "state eq 'FL'", "$limit": 10}
    r = requests.get(fema_base, params=params, headers=headers, timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   SUCCESS with User-Agent!")
except Exception as e:
    print(f"   Error: {type(e).__name__}")

# ==================== HURDAT2 INVESTIGATION ====================
print("\n" + "=" * 70)
print("HURDAT2 - Testing different approaches")
print("-" * 70)

hurdat2_urls = [
    ("Primary", "https://www.nhc.noaa.gov/data/hurdat2/hurdat2.txt"),
    ("Alternative 1", "https://www.nhc.noaa.gov/data/hurdat2.txt"),
    ("Alternative 2", "https://ftp.nhc.noaa.gov/atcf/archive/hurdat2.txt"),
    ("Alternative 3", "https://www.nhc.noaa.gov/data/hurdat2-1851-2023.txt"),
    ("Wayback Machine", "https://web.archive.org/web/20231201000000*/nhc.noaa.gov/data/hurdat2/*"),
]

print("\n1. Testing different HURDAT2 URLs:")
for name, url in hurdat2_urls:
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        print(f"   {name:20s}: HTTP {r.status_code} - {r.url}")

        if r.status_code == 200:
            # Try to get actual content
            r = requests.get(url, timeout=10)
            size_mb = len(r.text) / 1024 / 1024
            lines = len(r.text.split('\n'))
            print(f"      Content: {size_mb:.1f} MB, {lines:,} lines")
            print(f"      SUCCESS! First line: {r.text.split(chr(10))[0][:60]}")
            break
    except Exception as e:
        print(f"   {name:20s}: {type(e).__name__}")

# Try NOAA data portal
print("\n2. Checking NOAA data portal:")
try:
    r = requests.get("https://data.noaa.gov/api/3/action/package_search?q=hurdat", timeout=5)
    print(f"   NOAA API Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Found {len(data.get('result', {}).get('results', []))} HURDAT datasets")
except Exception as e:
    print(f"   Error: {type(e).__name__}")

# Try FTP access
print("\n3. FTP endpoint check:")
ftp_urls = [
    "ftp://ftp.nhc.noaa.gov/atcf/archive/hurdat2.txt",
    "ftp://ftp.noaa.gov/data/",
]
for url in ftp_urls:
    print(f"   {url}: (FTP - requires FTP client)")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
FEMA API:
  1. Try the base URL first with just $limit parameter
  2. If that works, add filters incrementally (state first, then dates)
  3. Check if authentication is needed (unlikely for public data)
  4. Monitor https://www.fema.gov/api/open/api/ for documentation

HURDAT2:
  1. Primary URL returning 404 - server may be down or file moved
  2. Try FTP access if HTTP doesn't work (ftp://ftp.nhc.noaa.gov/...)
  3. Check NOAA data portal for updated dataset location
  4. Use Wayback Machine if historical data needed

For your pipeline:
  - If FEMA works: Replace create_synthetic_fema_data() with live fetch
  - If HURDAT2 works: Replace create_synthetic_hurdat2() with live fetch
  - Otherwise: Continue with synthetic data (fully functional)
""")
