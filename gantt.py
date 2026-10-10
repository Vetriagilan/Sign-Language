import matplotlib.pyplot as plt
import numpy as np

# Define the project phases and their timelines
tasks = [
    "4. Real-Time Integration, UI Testing & Final Report",
    "3. MediaPipe Feature Extraction & AI Model Training",
    "2. Kaggle Dataset Acquisition & Custom OpenCV Recording",
    "1. Literature Survey & Environment Setup (Python 3.11)"
]

# (Start Week, Duration in Weeks)
# Note: Matplotlib plots from 0, so Week 1 starts at 0.
schedule = [
    (6, 2),  # Task 4 runs from start of week 7 to end of 8
    (4, 2),  # Task 3 runs from start of week 5 to end of 6
    (2, 2),  # Task 2 runs from start of week 3 to end of 4
    (0, 2)   # Task 1 runs from start of week 1 to end of 2
]

# Colors for the bars matching your PPT theme
colors = ['#800000', '#000080', '#008000', '#4B0082'] # Maroon, Navy, Green, Indigo

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 4))

# Plot each task as a horizontal bar
for i, (task, (start, duration)) in enumerate(zip(tasks, schedule)):
    ax.barh(task, duration, left=start, height=0.6, color=colors[i], edgecolor='black')

# Format the X-axis to represent weeks
ax.set_xticks(np.arange(0, 9, 1))
ax.set_xticklabels(['', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'])
ax.set_xlim(0, 8)

# Add gridlines and labels
ax.grid(axis='x', linestyle='--', alpha=0.7)
ax.set_xlabel('Project Timeline', fontsize=12, fontweight='bold', color='#000080')
ax.set_title('PROPOSED METHODOLOGY (Gantt Chart)', fontsize=16, fontweight='bold', color='#000080', pad=20)

# Adjust layout and save the image
plt.tight_layout()
output_filename = 'gantt_chart_slide10.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

print(f"Success! Gantt chart saved as '{output_filename}' in your project folder.")