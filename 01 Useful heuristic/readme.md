# Useful Heuristic

在本文件夹中，主要用来存储一些求解流水车间调度问题的启发式算法

## 1 Johnson Algorithm

**适用条件:** 两机台流水车间调度问题（改进的约翰逊算法可以适用于三机台的情况）

**基本思想：**

* **Step 1 分组:** 将所有任务根据在两个机台上的加工时间进行比较，并分成两组
  * A 组：在机器1上的加工时间小于等于机器2上的加工时间的任务
  * B 组：在机器1上的加工时间大于机器2上的加工时间的任务
* **Step 2 排序:** 在每个组内，对任务进行排序
  * 组 A 的任务按照在机器1上的加工时间从小到大排序
  * 组 B 的任务按照在机器1上的加工时间从大到小排序
* **Step 3 合并:** 将排序后的组A和组B的任务顺序合并，形成最终的任务加工顺序。如果组A中的任务数量小于或等于组B，则将组A的任务放在前面，组B的任务放在后面；如果组A中的任务数量多于组B，则将组B的任务放在前面，组A的任务放在后面
* **Step 4 计算 makespan:** 根据最终的任务加工顺序，计算两台机器的开始和结束时间，并确定makespan。



## 2 Modified Johnson Algorithm

**论文来源:**(COR 2019) [Three-machine flow shop scheduling with overlapping waiting time constraints](https://www.sciencedirect.com/science/article/pii/S0305054818301631)



## 3 NEH Algorithm

**算法来源:** NEH算法（Nawaz-Enscore-Ham Algorithm）是一种启发式算法，由Nawaz、Enscore和Ham在1983年提出，用于解决流水车间调度问题（Permutation Flow Shop Scheduling Problem, PFSP）中的Makespan最小化问题

**适用条件:** 多工件多机台的流水车间调度问题

**基本思想:**

* **Step 1 计算权重:** 对于每个作业，计算其在每台机器上的加工时间的总和，这个总和被称为作业的权重。

* **Step 2 排序：** 根据作业的权重进行排序。对于第一台机器，选择权重最小的作业放在序列的开始；对于最后一台机器，选择权重最大的作业放在序列的末尾。

* **Step 3 插入:** 对于剩余的作业，使用插入技术来确定它们在序列中的位置。插入时考虑的规则是最小化作业在机器上的空闲时间和部分完成时间。

* **Step 4 迭代:** 通过迭代过程，不断调整作业序列，直到找到一个局部最优解。
