import numpy as np
import matplotlib.pyplot as plt

# Data setup for each biomarker group and specified lower mean accuracy for random performance
groups = ['All biomarkers', 'Prostate biomarkers', 'Breast biomarkers', 'Ovarian biomarkers', 'Colorectal biomarkers', 'Pancreatic biomarkers']
random_means = [0.6, 0.33, 0.43, 0.51, 0.39, 0.52]

# Generate accuracies for each biomarker group with a specific mean and standard deviation
accuracies = {
    'All biomarkers': np.random.normal(loc=0.87, scale=0.05, size=30),
    'Prostate biomarkers': np.random.normal(loc=0.92, scale=0.02, size=30),
    'Breast biomarkers': np.random.normal(loc=0.88, scale=0.03, size=30),
    'Ovarian biomarkers': np.random.normal(loc=0.83, scale=0.06, size=30),
    'Colorectal biomarkers': np.random.normal(loc=0.82, scale=0.07, size=30),
    'Pancreatic biomarkers': np.random.normal(loc=0.81, scale=0.05, size=30)
}

# Generate random performance data with the same spread as each biomarker group but with lower mean accuracy
random_performance_adjusted = [
    np.random.normal(loc=mean, scale=accuracies[group].std(), size=30)
    for mean, group in zip(random_means, groups)
]

# Define unique colors for each biomarker group similar to previous image
colors = ['skyblue', 'lightgreen', 'lightcoral', 'khaki', 'mediumpurple', 'pink']

# Prepare data for a single box plot
data_to_plot = []
box_labels = []
positions = []

# Adjust positions for tighter grouping
position_index = 1
for group, random_acc_data in zip(groups, random_performance_adjusted):
    # Append cancer biomarker data and random data for each group
    data_to_plot.append(accuracies[group])
    data_to_plot.append(random_acc_data)
    box_labels.append(group)
    box_labels.append(f"Random set of\nvariants for {group.lower()}")
    positions.extend([position_index, position_index + 0.6])  # Tighter gap between pairs
    position_index += 2.5  # Reduced gap between groups

# Create a single box plot with all groups side by side
fig, ax = plt.subplots(figsize=(12, 6))

# Box plot configuration with adjusted positions
bp = ax.boxplot(data_to_plot, patch_artist=True, positions=positions, widths=0.5, showfliers=False)

# Set colors for box plots, alternating between colors for biomarkers and grey for random
for i, box in enumerate(bp['boxes']):
    if i % 2 == 0:
        box.set_facecolor(colors[i // 2])  # Cancer biomarker colors
    else:
        box.set_facecolor('grey')  # Random set color

# Customize medians only
for median in bp['medians']:
    median.set(color='#000000', linewidth=2)

# Adding labels
ax.set_xticks(positions)  # Add ticks for all positions
ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel('Prediction Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim(0.2, 1.0)
ax.grid(True, linestyle='--')

# Adjust layout for better readability
plt.tight_layout()
plt.show()
