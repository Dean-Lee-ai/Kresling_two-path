import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义函数 f(x, y, z)
def f(x, y, z):
    return x**2 + y**2 + z**2  # 示例函数

# 定义约束方程 g1(x, y, z) = 0 和 g2(x, y, z) = 0
def g1(x, y, z):
    return x + y + z - 1  # 示例约束

def g2(x, y, z):
    return x**2 + y**2 - z  # 示例约束

# 定义方程组，用于求解约束
def equations(vars):
    x, y, z = vars
    return [g1(x, y, z), g2(x, y, z), 0]  # 添加第三个方程以匹配形状

# 初始猜测值
initial_guess = [0.5, 0.5, 0.5]

# 使用 fsolve 求解约束方程
solution = fsolve(equations, initial_guess)
x_sol, y_sol, z_sol = solution

print(f"Solution: x = {x_sol}, y = {y_sol}, z = {z_sol}")

# 计算 f(x, y, z) 的值
f_value = f(x_sol, y_sol, z_sol)
print(f"f(x, y, z) = {f_value}")

# 绘制图像
# 假设 x 是自变量，绘制 f(x) 的图像
x_values = np.linspace(-10, 10, 100)
f_values = []

for x in x_values:
    def equations_with_x(vars):
        y, z = vars
        return [g1(x, y, z), g2(x, y, z)]  # 两个方程，两个变量
    
    solution_with_x = fsolve(equations_with_x, [0.5, 0.5])
    y_sol, z_sol = solution_with_x
    f_values.append(f(x, y_sol, z_sol))

plt.plot(x_values, f_values)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) under constraints')
plt.show()
