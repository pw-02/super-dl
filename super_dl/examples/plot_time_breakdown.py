import matplotlib.pyplot as plt
import numpy as np

def plot_time_breakdown(num_epochs, pytorch_data_fetch_time, pytorch_data_transform_time,
                        super_data_fetch_time, super_data_transform_time, processing_time, save_path=None):
    # Calculate times in minutes with a range of 1-5 minutes
    pytorch_data_fetch_time = np.random.uniform(1, 5, num_epochs)
    pytorch_data_transform_time = np.random.uniform(1, 5, num_epochs)
    
    # Adjust SUPER DataLoader times for consistent throughput and reduced data fetch time
    super_data_fetch_time = np.random.uniform(0.5, 1.5, num_epochs)  # Reduced data fetch time
    super_data_transform_time = np.random.uniform(1, 5, num_epochs)  # Adjusted transform time
    
    processing_time = np.random.uniform(1, 5, num_epochs)

    # Calculate total times
    total_pytorch_data_loading_time = pytorch_data_fetch_time + pytorch_data_transform_time + processing_time
    total_super_data_loading_time = super_data_fetch_time + super_data_transform_time + processing_time

    # Plotting the results for Time Breakdown
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(1, num_epochs + 1)

    color_scheme_pytorch = ['#3182bd', '#6baed6', '#9ecae1']
    color_scheme_super = ['#e6550d', '#fd8d3c', '#fdae6b']

    # PyTorch DataLoader bars
    ax.bar(index, pytorch_data_fetch_time[:num_epochs], bar_width, label='Data Fetch (PyTorch)',
           color=color_scheme_pytorch[0], edgecolor='black')
    ax.bar(index, pytorch_data_transform_time[:num_epochs], bar_width, label='Data Transform (PyTorch)',
           bottom=pytorch_data_fetch_time[:num_epochs], color=color_scheme_pytorch[1], edgecolor='black')
    ax.bar(index, processing_time[:num_epochs], bar_width, label='Processing (PyTorch)',
           bottom=(pytorch_data_fetch_time + pytorch_data_transform_time)[:num_epochs],
           color=color_scheme_pytorch[2], edgecolor='black')

    # SUPER DataLoader bars stacked on top
    ax.bar(index + bar_width, super_data_fetch_time[:num_epochs], bar_width, label='Data Fetch (SUPER)',
           color=color_scheme_super[0], alpha=0.7, edgecolor='black')
    ax.bar(index + bar_width, super_data_transform_time[:num_epochs], bar_width, label='Data Transform (SUPER)',
           bottom=super_data_fetch_time[:num_epochs], color=color_scheme_super[1], alpha=0.7, edgecolor='black')
    ax.bar(index + bar_width, processing_time[:num_epochs], bar_width, label='Processing (SUPER)',
           bottom=(super_data_fetch_time + super_data_transform_time)[:num_epochs],
           color=color_scheme_super[2], alpha=0.7, edgecolor='black')

    ax.set_ylabel('Time (minutes)')  # Adjusted the y-axis label
    ax.set_xlabel('Epoch')  # Added the x-axis label
    ax.set_title('Time Breakdown')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='small')  # Adjust the legend position below the figure and increase font size

    # Explicitly set the x-axis ticks to integer values
    plt.xticks(np.arange(1, num_epochs + 1, 1))

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path + "_time_breakdown.png")

    plt.show()

def plot_throughput_comparison(num_epochs, pytorch_throughput, super_throughput, save_path=None):
    # Plotting the results for Throughput Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = np.arange(1, num_epochs + 1)

    color_scheme_pytorch = ['#3182bd']
    color_scheme_super = ['#e6550d']

    ax.bar(index, pytorch_throughput[:num_epochs] * 100, bar_width, label='PyTorch Dataloader',
           color=color_scheme_pytorch[0], edgecolor='black')
    ax.bar(index + bar_width, super_throughput[:num_epochs] * 100, bar_width, label='SUPER Dataloader',
           color=color_scheme_super[0], alpha=0.7, edgecolor='black')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Throughput (Batches/Second)')  # Adjusted the y-axis label
    ax.set_title('Throughput Comparison Over {} Epochs'.format(num_epochs))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')  # Adjust the legend position below the figure and increase font size

    # Explicitly set the x-axis ticks to integer values
    plt.xticks(np.arange(1, num_epochs + 1, 1))

    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path + "_throughput_comparison.png")

    plt.show()

if __name__ == "__main__":
    # Simulated data for demonstration with a range of 1-5 minutes for time
    num_epochs = 4

    # PyTorch DataLoader times with a range of 1-5 minutes
    pytorch_data_fetch_time = np.random.uniform(1, 5, num_epochs)
    pytorch_data_transform_time = np.random.uniform(1, 5, num_epochs)

    # SUPER DataLoader times with a range of 1-5 minutes
    super_data_fetch_time = np.random.uniform(0.5, 1.5, num_epochs)  # Reduced data fetch time
    super_data_transform_time = np.random.uniform(1, 5, num_epochs)

    processing_time = np.random.uniform(1, 5, num_epochs)

    # Calculate total times
    total_pytorch_data_loading_time = pytorch_data_fetch_time + pytorch_data_transform_time + processing_time
    total_super_data_loading_time = super_data_fetch_time + super_data_transform_time + processing_time

    # Calculate throughput
    pytorch_throughput = 1 / total_pytorch_data_loading_time
    super_throughput = 1 / total_super_data_loading_time

    # Save figures to files
    plot_time_breakdown(num_epochs, pytorch_data_fetch_time, pytorch_data_transform_time,
                        super_data_fetch_time, super_data_transform_time, processing_time, save_path="figure1")
    plot_throughput_comparison(num_epochs, pytorch_throughput, super_throughput, save_path="figure2")
