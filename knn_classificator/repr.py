import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import alpha

from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../water_potability.csv')
data = data.dropna()

### Data visualization ###
# Parameter distribution
fig = plt.figure(figsize=(16, 12))
for n, col in enumerate(data.columns[:-1], 1):
    plt.subplot(3,3,n)
    plt.hist(data[data['Potability']==0][col], alpha=0.6, label='Not potable', bins=30)
    plt.hist(data[data['Potability'] == 1][col], alpha=0.6, label='Potable', bins=30)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
plt.savefig('figs/distributions.png', dpi=150)
plt.close()

# Comparative scatter
fig = plt.figure(figsize=(8, 6))
potable = data[data['Potability'] == 1]
non_potable = data[data['Potability'] == 0]
plt.scatter(
    non_potable['ph'],
    non_potable['Hardness'],
    alpha=0.5,
    label='Not potable',
    s=20
)
plt.scatter(
    potable['ph'],
    potable['Hardness'],
    alpha=0.5,
    label='Potable',
    s=20
)
plt.xlabel('pH')
plt.ylabel('Hardness')
plt.legend()
plt.title('pH vs. Hardness')
plt.tight_layout()
plt.savefig('figs/ph_vs_hardness.png', dpi=150)
plt.close()

# Correlation heatmap
fig = plt.figure(figsize=(8, 6))
cor = data.corr()
ims = plt.imshow(cor, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(ims)
plt.xticks(range(len(cor.columns)), cor.columns, rotation=45, ha='right')
plt.yticks(range(len(cor.columns)), cor.columns)
for idx in range(len(cor.columns)):
    for sub in range(len(cor.columns)):
        plt.text(sub, idx, s=f'{cor.iloc[idx, sub]:.2f}', ha='center', va='center', fontsize=6)
plt.title('Correlation heatmap')
plt.tight_layout()
plt.savefig('figs/correlation_heatmap.png', dpi=150)
plt.close()

# Density
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
for n, col in enumerate(data.columns[:-1]):
    ax = axs[n//3, n%3]
    data[data['Potability']==0][col].plot(kind='density', ax=ax, label="Not potable", alpha=0.6)
    data[data['Potability']==1][col].plot(kind='density', ax=ax, label="Potable", alpha=0.6)
    ax.set_xlabel(col)
    ax.legend()
plt.tight_layout()
plt.savefig('figs/density.png', dpi=150)
plt.close()

# Pairing on correlated features for potability
p_cor = data.corr()['Potability'].abs().sort_values(ascending=False)
top = p_cor.index[1:5]
fig, axs = plt.subplots(4,4, figsize=(14,14))
for i, ff in enumerate(top):
    for j, sf in enumerate(top):
        ax = axs[i, j]
        if i == j:
            ax.hist(data[data['Potability']==0][ff], alpha=0.6, bins=20)
            ax.hist(data[data['Potability']==1][ff], alpha=0.6, bins=20)
        else:
            ax.scatter(data[data['Potability']==0][sf], data[data['Potability']==0][ff], alpha=0.3, s=5)
            ax.scatter(data[data['Potability']==1][sf], data[data['Potability']==1][ff], alpha=0.3, s=5)

        if i == 3:
            ax.set_xlabel(sf, fontsize=6)
        if j == 0:
            ax.set_ylabel(ff, fontsize=6)
plt.tight_layout()
plt.savefig('figs/pairing.png', dpi=150)
plt.close()


### KNN Analisys
def benchmark(k_values, acc):
    plt.figure(figsize=(12, 6))
    plt.subplot(1,2,1)
    plt.plot(k_values, acc, marker='o', linewidth=2, markersize=5)
    plt.xlabel('K', fontsize=14)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Impact of K in accuracy')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('figs/benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()


### Machine Learning
# Feature importance
rndf = RandomForestClassifier(n_estimators=100, random_state=42)
