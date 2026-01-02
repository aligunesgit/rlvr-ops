import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_core = '#3498db'
color_training = '#e74c3c'
color_rewards = '#2ecc71'
color_deploy = '#f39c12'
color_monitor = '#9b59b6'

def draw_box(ax, x, y, width, height, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black',
                         facecolor=color,
                         alpha=0.7,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight='bold',
           color='white')

def draw_arrow(ax, x1, y1, x2, y2, style='->'):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color='black',
                           linewidth=2,
                           mutation_scale=20)
    ax.add_patch(arrow)

# Title
ax.text(5, 9.5, 'RLVR-Ops Framework Architecture',
       ha='center', fontsize=18, fontweight='bold')

# Top Layer
draw_box(ax, 0.5, 7.5, 2, 1.5, 'Core Module\n\n• Policy\n• Agent\n• Environment', color_core, 9)
draw_box(ax, 3.5, 7.5, 2, 1.5, 'Training Module\n\n• GRPO\n• Engine\n• Trainer', color_training, 9)
draw_box(ax, 6.5, 7.5, 2, 1.5, 'Rewards Library\n\n• Exact Match\n• Code Exec\n• F1 Score', color_rewards, 9)

draw_arrow(ax, 2.5, 8.25, 3.5, 8.25)
draw_arrow(ax, 5.5, 8.25, 6.5, 8.25)

# Middle Layer
draw_box(ax, 1.5, 5, 6, 1.8, '', '#ecf0f1')
ax.text(4.5, 6.5, 'Training Loop', ha='center', fontsize=12, fontweight='bold')

for i in range(4):
    x_pos = 2 + i * 1.4
    draw_box(ax, x_pos, 5.3, 0.9, 0.5, f'Roll-{i+1}', '#3498db', 8)
    draw_box(ax, x_pos, 4.7, 0.9, 0.4, f'R={i%2}', '#2ecc71', 7)

draw_box(ax, 2.5, 3.8, 2, 0.6, 'Advantage\nComputation', '#f39c12', 9)
draw_arrow(ax, 4.5, 5, 4.5, 4.4)

draw_box(ax, 5, 3.8, 2, 0.6, 'Policy\nUpdate', '#e74c3c', 9)
draw_arrow(ax, 4.5, 3.8, 5, 4.1)

# Bottom Layer
draw_box(ax, 0.5, 1.5, 2, 1.5, 'Deployment\n\n• FastAPI\n• Docker\n• Kubernetes', color_deploy, 9)
draw_box(ax, 3.5, 1.5, 2, 1.5, 'Monitoring\n\n• Metrics\n• Logging\n• Grafana', color_monitor, 9)
draw_box(ax, 6.5, 1.5, 2, 1.5, 'Optimization\n\n• Rollout Pruning\n• Batching\n• Caching', color_core, 9)

draw_arrow(ax, 2.5, 5, 1.5, 3)
draw_arrow(ax, 4.5, 5, 4.5, 3)
draw_arrow(ax, 6.5, 5, 7.5, 3)

plt.tight_layout()
plt.savefig('figure1_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1 saved: figure1_architecture.png")
plt.close()
