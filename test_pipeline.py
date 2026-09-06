"""Quick test of pipeline components."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Test 1: Create synthetic data
print("Test 1: Creating synthetic data...")
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'claimNumber': [f'CLAIM{i:08d}' for i in range(n)],
    'latitude': np.random.uniform(24.5, 30.5, n),
    'longitude': np.random.uniform(-87.5, -80.0, n),
    'amountPaid': np.random.exponential(50000, n),
    'dateOfLoss': pd.date_range('2020-01-01', periods=n, freq='D'),
})
print(f"  Created {len(df)} rows")

# Test 2: Feature engineering
print("\nTest 2: Feature engineering...")
df['log_amountpaid'] = np.log1p(df['amountPaid'])
print(f"  Log transform: min={df['log_amountpaid'].min():.2f}, max={df['log_amountpaid'].max():.2f}")

# Test 3: Create a plot
print("\nTest 3: Creating a test plot...")
plt.figure(figsize=(12, 6))
plt.hist(df['log_amountpaid'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Log Amount Paid')
plt.ylabel('Frequency')
plt.title('Test Distribution Plot')
plt.tight_layout()
plt.savefig('C:\\Users\\Brian\\Desktop\\Claude Projects\\Insurance Project\\test_plot.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Plot saved: test_plot.png")

print("\n✓ All tests passed!")
