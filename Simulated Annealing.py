import numpy as np
import random
import math
import matplotlib.pyplot as plt

# 能量函数定义
def energy(x, y, k1=1.0, k2=1.0, x0=0.0, y0=0.0, mu=1.0):
    elastic_energy = 0.5 * k1 * (x - x0)**2 + 0.5 * k2 * (y - y0)**2
    magnetic_energy = 0.5 * mu * (x**2 + y**2)
    return elastic_energy + magnetic_energy

# 邻域解生成函数
def get_neighbor(x, y, step_size=0.1):
    new_x = x + random.uniform(-step_size, step_size)
    new_y = y + random.uniform(-step_size, step_size)
    return new_x, new_y

# 模拟退火算法
def simulated_annealing(initial_x, initial_y, initial_temp, temp_decay, max_iterations):
    x, y = initial_x, initial_y
    current_energy = energy(x, y)
    best_x, best_y = x, y
    best_energy = current_energy

    # 记录能量随时间变化的轨迹
    energy_history = [(x, y, current_energy)]

    for iteration in range(max_iterations):
        temp = initial_temp * (temp_decay ** iteration)
        
        # 生成邻域解
        new_x, new_y = get_neighbor(x, y)
        new_energy = energy(new_x, new_y)
        
        # 判断是否接受新解
        if new_energy < current_energy:
            # 如果新解能量较低，直接接受
            x, y = new_x, new_y
            current_energy = new_energy
        else:
            # 否则以一定概率接受较差解
            prob = math.exp((current_energy - new_energy) / temp)
            if random.random() < prob:
                x, y = new_x, new_y
                current_energy = new_energy
        
        # 更新最优解
        if current_energy < best_energy:
            best_x, best_y = x, y
            best_energy = current_energy
        
        # 记录能量和位置
        energy_history.append((x, y, current_energy))

    return best_x, best_y, best_energy, energy_history

# 参数设置
initial_x, initial_y = 2.0, 2.0    # 初始位置
initial_temp = 100.0                # 初始温度
temp_decay = 0.99                   # 温度衰减系数
max_iterations = 1000               # 最大迭代次数

# 调用模拟退火算法
best_x, best_y, best_energy, energy_history = simulated_annealing(
    initial_x, initial_y, initial_temp, temp_decay, max_iterations
)

# 打印最终结果
print(f"最优解：x = {best_x}, y = {best_y}, 最优能量 = {best_energy}")

# 绘制能量轨迹
energy_history = np.array(energy_history)
plt.plot(energy_history[:, 2], label='Energy')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('Energy History')
plt.legend()
plt.show()
