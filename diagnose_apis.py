"""
Diagnose FEMA API and HURDAT2 endpoint issues
"""

import requests
import json

print("API ENDPOINT DIAGNOSTICS")
print("=" * 70)

# ==================== FEMA API INVESTIGATION ====================
print("\nFEMA API - Issue: HTTP 400 'Invalid version format'")
print("-" * 70)

fema_url = "https://www.fema.gov/api/open/data/FimaNfipClaims"

print(f"\n1. Testing base URL (no params):")
try:
    response = requests.get(fema_url, timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('content-type', 'N/A')}")
    if response.status_code == 200:
        print(f"   Response sample: {response.text[:200]}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n2. Testing with simplified filter (no datetime):")
try:
    params = {
        "$filter": "state eq 'FL'",
        "$limit": 10,
        "$offset": 0
    }
    response = requests.get(fema_url, params=params, timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text[:300]}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n3. Testing with dateOfLoss filter (different format):")
try:
    params = {
        "$filter": "state eq 'FL' and dateOfLoss gt 1978-01-01",
        "$limit": 10
    }
    response = requests.get(fema_url, params=params, timeout=5)
    print(f"   Status: {response.status_code}")
    if response.status_code != 200:
        print(f"   Error: {response.text[:300]}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n4. Checking FEMA API documentation/metadata endpoints:")
try:
    # Try metadata endpoint
    meta_url = "https://www.fema.gov/api/open/data"
    response = requests.get(meta_url, timeout=5)
    print(f"   Metadata endpoint status: {response.status_code}")
    if response.status_code == 200:
        data = response.json() if 'application/json' in response.headers.get('content-type', '') else None
        if data:
            print(f"   Available endpoints: {len(data) if isinstance(data, list) else 'dict'}")
            if isinstance(data, list) and len(data) > 0:
                print(f"   Sample: {data[0]}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n5. Trying alternative FEMA endpoints:")
alternatives = [
    "https://www.fema.gov/api/open/v1/FimaNfipClaims",
    "https://www.fema.gov/api/open/v2/FimaNfipClaims",
    "https://opendata.fema.gov/api/rest/v1/FimaNfipClaims",
]

for alt_url in alternatives:
    try:
        response = requests.get(alt_url, timeout=5)
        print(f"   {alt_url.split('/')[-2]}: HTTP {response.status_code}")
    except Exception as e:
        print(f"   {alt_url}: {type(e).__name__}")

# ==================== HURDAT2 INVESTIGATION ====================
print("\n" + "=" * 70)
print("HURDAT2 - Issue: HTTP 404 Not Found")
print("-" * 70)

print(f"\n1. Testing current URL:")
try:
    url = "https://www.nhc.noaa.gov/data/hurdat2/hurdat2.txt"
    response = requests.get(url, timeout=10, allow_redirects=True)
    print(f"   Status: {response.status_code}")
    print(f"   URL (after redirects): {response.url}")
    if response.status_code == 200:
        print(f"   File size: {len(response.text)} bytes")
        print(f"   First line: {response.text.split(chr(10))[0][:80]}")
except Exception as e:
    print(f"   Error: {e}")

print(f"\n2. Trying alternative HURDAT2 URLs:")
alternatives_hurdat2 = [
    "https://www.nhc.noaa.gov/data/hurdat2-1851-2023-041124.txt",
    "https://www.nhc.noaa.gov/data/hurdat2/hurdat2.txt/",
    "https://ftp.nhc.noaa.gov/atcf/archive/hurdat2.txt",
    "https://www.hurricanehunters.com/hurdat2.txt",
]

for alt_url in alternatives_hurdat2:
    try:
        response = requests.head(alt_url, timeout=5, allow_redirects=True)
        print(f"   {alt_url.split('/')[-1]}: HTTP {response.status_code}")
    except Exception as e:
        print(f"   {alt_url.split('/')[-1]}: {type(e).__name__}")

print(f"\n3. Checking NOAA base domains:")
noaa_bases = [
    "https://www.nhc.noaa.gov/data/",
    "https://ftp.nhc.noaa.gov/",
    "https://data.noaa.gov/",
]

for base in noaa_bases:
    try:
        response = requests.head(base, timeout=5, allow_redirects=True)
        print(f"   {base}: HTTP {response.status_code}")
    except Exception as e:
        print(f"   {base}: {type(e).__name__}")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
print("""
FEMA API Issue:
  - Error: "Invalid version format"
  - Likely cause: OData filter syntax or datetime format
  - Solution: Try alternative filter formats or check API versioning

HURDAT2 Issue:
  - Error: 404 Not Found
  - Likely cause: File temporarily moved or URL structure changed
  - Solution: Try alternative NOAA URLs or FTP endpoints

Recommendation:
  - For now, continue using synthetic data (already working)
  - Monitor FEMA and NOAA endpoints for restoration
  - When APIs return, re-run pipeline for real data
  - Fallback mechanism in pipeline handles both scenarios
""")
