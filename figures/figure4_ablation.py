import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

k_values = [1, 2, 4, 8]
k_accuracy = [4.0, 5.0, 6.0, 7.0]

ax1.plot(k_values, k_accuracy, marker='o', markersize=12, 
         linewidth=3, color='#3498db', label='Accuracy')
ax1.fill_between(k_values, k_accuracy, alpha=0.3, color='#3498db')
ax1.set_xlabel('Number of Rollouts (k)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Impact of Rollout Count', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(k_values)
ax1.set_ylim(3, 8)

for x, y in zip(k_values, k_accuracy):
    ax1.annotate(f'{y}%', xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center',
                fontsize=11, fontweight='bold')

temps = [0.5, 0.7, 0.9, 1.0, 1.2]
temp_accuracy = [4.5, 5.5, 6.0, 5.8, 5.0]

bars = ax2.bar(range(len(temps)), temp_accuracy, 
               color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6'],
               alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_xlabel('Temperature', fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Impact of Temperature', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(len(temps)))
ax2.set_xticklabels(temps)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(3, 8)

for bar, acc in zip(bars, temp_accuracy):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.15,
            f'{acc}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

optimal_idx = temp_accuracy.index(max(temp_accuracy))
bars[optimal_idx].set_edgecolor('gold')
bars[optimal_idx].set_linewidth(4)

plt.tight_layout()
plt.savefig('figure4_ablation.png', dpi=300, bbox_inches='tight')
print("✅ Figure 4 saved: figure4_ablation.png")
plt.close()
