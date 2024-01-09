import matplotlib.pyplot as plt
import numpy as np

# Simulated data for different Data Loaders - Training speeds for multiple GPT-2 models
num_models = 5
pytorch_dataloader_speeds = np.random.uniform(50, 150, size=num_models)
super_dataloader_speeds = np.random.uniform(80, 200, size=num_models)
quiver_dataloader_speeds = np.random.uniform(60, 180, size=num_models)
coordl_dataloader_speeds = np.random.uniform(70, 160, size=num_models)

# Plotting the figure
plt.figure(figsize=(12, 6))

bar_width = 0.2
bar_positions = np.arange(num_models)

plt.bar(bar_positions - 1.5*bar_width, pytorch_dataloader_speeds, width=bar_width, label='GPT-2 with PyTorch DataLoader', color='skyblue', edgecolor='black')
plt.bar(bar_positions - 0.5*bar_width, super_dataloader_speeds, width=bar_width, label='GPT-2 with SUPER DataLoader', color='lightcoral', edgecolor='black')
plt.bar(bar_positions + 0.5*bar_width, quiver_dataloader_speeds, width=bar_width, label='GPT-2 with Quiver DataLoader', color='lightgreen', edgecolor='black')
plt.bar(bar_positions + 1.5*bar_width, coordl_dataloader_speeds, width=bar_width, label='GPT-2 with CoorDL DataLoader', color='lightyellow', edgecolor='black')

plt.title('Training Speeds for Multiple GPT-2 Models: Various Data Loaders')
plt.xlabel('GPT-2 Model Variants')
plt.ylabel('Training Speed (batches per second)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add values on top of each bar
for i, pos in enumerate(bar_positions):
    plt.text(pos - 1.5*bar_width, pytorch_dataloader_speeds[i] + 3, f'{pytorch_dataloader_speeds[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')
    plt.text(pos - 0.5*bar_width, super_dataloader_speeds[i] + 3, f'{super_dataloader_speeds[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')
    plt.text(pos + 0.5*bar_width, quiver_dataloader_speeds[i] + 3, f'{quiver_dataloader_speeds[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')
    plt.text(pos + 1.5*bar_width, coordl_dataloader_speeds[i] + 3, f'{coordl_dataloader_speeds[i]:.2f}', ha='center', va='bottom', fontsize=8, color='black')

plt.tight_layout()

# Save the figure
plt.savefig('gpt2_training_speeds.png')
plt.show()
