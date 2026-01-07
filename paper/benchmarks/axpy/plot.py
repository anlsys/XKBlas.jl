import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <csv_file>")
    sys.exit(1)

# Read the CSV file with colon delimiter
df = pd.read_csv(sys.argv[1], sep=':')

# Convert time from seconds to milliseconds
df['time'] = df['time']
df = df[df['n'] >= 100000]

# Group by ngpus and n, calculate mean and std of time
grouped = df.groupby(['ngpus', 'n'])['time'].agg(['mean', 'std']).reset_index()

# Get unique ngpus values for different curves
unique_ngpus = sorted(grouped['ngpus'].unique())

# Create the plot with larger font sizes
plt.figure(figsize=(12, 7))
plt.rcParams.update({'font.size': 16})

# Different markers and line styles for each curve
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

# Set log scales first
plt.xscale("log", base=2)
plt.yscale("log", base=10)

for idx, ngpu in enumerate(unique_ngpus):
    data = grouped[grouped['ngpus'] == ngpu]
    plt.errorbar(data['n'], data['mean'], yerr=data['std'],
                 marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 label=f'{ngpu} GPUs', capsize=5, linewidth=2, markersize=8)

plt.xlabel('Number of elements per vector', fontsize=20)
plt.ylabel('Time (s)', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, alpha=0.3, which='both')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlim(left=df['n'].min() / 1.25, right=df['n'].max() * 1.25)
plt.ylim(top=0.01)

plt.tight_layout()
plt.show()
