# -*- coding: utf-8 -*-
"""
Author: Long Chen

email:lchen1307@mail.ustc.edu.cn
Date: Sat Aug 31 14:50:54 2024
University of Science and Technology of China
"""

import matplotlib.pyplot as plt

def plot_gantt_chart(tasks, job_colors=None, machine_labels=None, title='Gantt Chart', xlabel='Time',
                     ylabel='Machine ID', save_path=None):
    """
    Plots a Gantt chart for the given tasks.

    Parameters:
    - tasks: List of tuples in the format (job number, machine number, start time, end time)
    - job_colors: Dictionary mapping job numbers to colors (default: None, auto color assignment)
    - machine_labels: Dictionary mapping machine numbers to labels (default: None, uses 'Machine X' format)
    - title: Title of the Gantt chart (default: 'Gantt Chart')
    - xlabel: Label for the x-axis (default: 'Time')
    - ylabel: Label for the y-axis (default: 'Machine ID')
    - save_path: If provided, saves the chart to the given path (default: None, does not save)
    """

    # Create a color map for jobs if not provided
    if job_colors is None:
        job_colors = {}
        # Generate default colors for jobs if not provided
        default_colors = ['#72c3a3', '#a5c2e2', '#fea040', '#fdd835', '#ff7043']
        for i, job_id in enumerate(set([task[0] for task in tasks])):
            job_colors[job_id] = default_colors[i % len(default_colors)]

    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Draw a bar for each task
    for task in tasks:
        job_id, machine_id, start_time, end_time = task
        color = job_colors.get(job_id, 'grey')  # Get color for job, default to grey
        ax.barh(y=machine_id, width=end_time - start_time, left=start_time, color=color, edgecolor='black',
                label=f'Job {job_id}' if job_id not in ax.get_legend_handles_labels()[1] else "")

    # Set the chart title and axis labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set y-axis tick labels
    if machine_labels is None:
        # Default machine labels: Machine 1, Machine 2, ...
        machine_labels = {machine_id: f'Machine {machine_id}' for machine_id in set([task[1] for task in tasks])}

    ax.set_yticks(list(machine_labels.keys()))
    ax.set_yticklabels([machine_labels[i] for i in ax.get_yticks()])

    # Set the range for the x-axis
    ax.set_xlim(0, max([task[3] for task in tasks]) + 1)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Display the chart
    plt.tight_layout()

    # Save the chart if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, format='png', bbox_inches='tight', dpi=800)

    # Show the chart
    plt.show()


# Example usage of the function

# tasks = [
#     (1, 1, 0, 2),
#     (1, 2, 2, 5),
#     (2, 1, 4, 7),
#     (2, 2, 7, 9)
# ]
#
# job_colors = {
#     1: '#72c3a3',  # Color for Job 1
#     2: '#a5c2e2',  # Color for Job 2
# }
#
# machine_labels = {1: 'Machine 1', 2: 'Machine 2'}
#
# # Call the function to plot the Gantt chart
# plot_gantt_chart(tasks, job_colors=job_colors, machine_labels=machine_labels, title='Gantt Chart Example',
#                  save_path='gantt_chart_example.png')
