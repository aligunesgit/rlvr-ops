import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Random', 'Zero-shot\nGPT-2', 'RLVR-Ops\nk=1', 'RLVR-Ops\nk=4', 'RLVR-Ops\nk=8']
accuracy = [0, 0, 4, 6, 7]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#27ae60', '#1e8449']

bars = ax.bar(methods, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

for bar, acc in zip(bars, accuracy):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{acc}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Baseline Comparison on GSM8k', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure3_results.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3 saved: figure3_results.png")
plt.close()
