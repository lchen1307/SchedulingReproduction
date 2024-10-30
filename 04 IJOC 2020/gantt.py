import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_gantt_chart(schedule):
    """
    绘制平行机调度甘特图（每个工件只在一台机台上加工，不同工件使用不同颜色，并在色块上标注工件名称）
    :param schedule: 字典，格式为 {工件编号: (机台编号, 开始时间, 结束时间), ...}
    """
    # 获取所有机台编号
    machines = set(machine for machine, _, _ in schedule.values())
    machines = sorted(list(machines))

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 为每个机台分配一个y轴位置
    machine_positions = {machine: i for i, machine in enumerate(machines)}

    # 为每个工件分配一个颜色
    job_colors = plt.cm.tab20(np.linspace(0, 1, len(schedule)))

    # 绘制甘特图
    legend_handles = []
    for i, (job, (machine, start, end)) in enumerate(schedule.items()):
        machine_pos = machine_positions[machine]
        color = job_colors[i]  # 为每个工件分配颜色
        ax.broken_barh([(start, end - start)], (machine_pos - 0.4, 0.8),
                       facecolors=color, edgecolor='black', label=job)

        # 在色块上标注工件名称
        ax.text((start + end) / 2, machine_pos, job, ha='center', va='center', color='black', fontsize=10)

        legend_handles.append(mpatches.Patch(color=color, label=job))

    # 设置y轴刻度和标签
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines)
    ax.set_ylabel('Machine')

    # 设置x轴刻度和标签
    ax.set_xlabel('Time')
    ax.set_title('Parallel Machine Scheduling Gantt Chart')

    # 添加图例
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 显示图形
    plt.tight_layout()
    plt.show()


# 示例输入数据
schedule = {
    'Job1': ('Machine1', 0, 6),
    'Job2': ('Machine1', 6, 10),
    'Job3': ('Machine1', 10, 13),
    'Job4': ('Machine1', 13, 19),
    'Job5': ('Machine2', 0, 5),
    'Job6': ('Machine2', 5, 11),
    'Job7': ('Machine2', 11, 15),
    'Job8': ('Machine2', 15, 23)
}

# 调用函数绘制甘特图
draw_gantt_chart(schedule)