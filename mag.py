import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

miu0 = 4*np.pi*10**-7
mu0 = 4*np.pi*10**-7
# 计算矢量差
def rij(i, j, pts):
    return pts[j] - pts[i]

# 磁力计算
def Fmag(mi, mj, rijVec, miu0):
    norm_rij = np.linalg.norm(rijVec)  # 计算rijVec的模
    dot_mi_rij = np.dot(mi, rijVec)  # 计算mi与rijVec的点积
    dot_mj_rij = np.dot(mj, rijVec)  # 计算mj与rijVec的点积
    dot_mi_mj = np.dot(mi, mj)  # 计算mi与mj的点积

    # 计算磁力
    term1 = dot_mi_rij * mj + dot_mj_rij * mi + dot_mi_mj * rijVec
    term2 = 5 * (dot_mi_rij * dot_mj_rij / norm_rij**2) * rijVec
    
    return -3 * miu0 / (4 * np.pi * norm_rij**5) * (term1 - term2)

# 磁能量计算
def Emag(mi, mj, rijVec, miu0):
    norm_rij = np.linalg.norm(rijVec)  # 计算rijVec的模
    dot_mi_mi = np.dot(mi, mi)  # 计算mi与mi的点积
    dot_mi_rij = np.dot(mi, rijVec)  # 计算mi与rijVec的点积
    dot_mj_rij = np.dot(mj, rijVec)  # 计算mj与rijVec的点积
    
    return miu0 / (4 * np.pi) * (dot_mi_mi / norm_rij**3 - 3 * (dot_mi_rij * dot_mj_rij) / norm_rij**5)

# 弹簧几何量
def c(hs, phi, r, R, theta_p0):
    return np.sqrt(hs**2 + r**2 + R**2 - 2 * R * r * np.cos(theta_p0 + phi))

def d(hs, phi, r, R, theta_p0, n):
    return np.sqrt(hs**2 + r**2 + R**2 - 2 * R * r * np.cos(theta_p0 + phi + 2 * np.pi / n))

# 弹簧力
def Fcc(hs, phi, r, R, theta_p0, hp0, kc):
    return kc * (c(hs, phi, r, R, theta_p0) - c(hp0, 0, r, R, theta_p0))

def Fdd(hs, phi, r, R, theta_p0, hp0, kd, n):
    return kd * (d(hs, phi, r, R, theta_p0, n) - d(hp0, 0, r, R, theta_p0, n))

# 力和力矩的角度量
def Ac(hs, phi, R, r, theta_p0):
    return np.arctan(np.sqrt(R**2 + r**2 - 2 * R * r * np.cos(theta_p0 + phi)) / hs)

def Ad(hs, phi, R, r, theta_p0, n):
    return np.arctan(np.sqrt(R**2 + r**2 - 2 * R * r * np.cos(theta_p0 + phi + (2 * np.pi) / n)) / hs)

# Zeta 函数
def zeta(x, R, r):
    return np.sin(x) / np.sqrt(R**2 + r**2 - 2 * R * r * np.cos(x))

# 总力函数
def Ftotal(hs, phi, r, R, theta_p0, hp0, kc, kd, n):
    return n * (Fcc(hs, phi, r, R, theta_p0, hp0, kc) * np.cos(Ac(hs, phi, R, r, theta_p0)) +
                Fdd(hs, phi, r, R, theta_p0, hp0, kd, n) * np.cos(Ad(hs, phi, R, r, theta_p0, n)))

# 总力矩函数
def Ttotal(hs, phi, r, R, theta_p0, hp0, kc, kd, n):
    return n * R * r * (Fcc(hs, phi, r, R, theta_p0, hp0, kc) * np.sin(Ac(hs, phi, R, r, theta_p0)) * 
                        zeta(theta_p0 + phi, R, r) + 
                        Fdd(hs, phi, r, R, theta_p0, hp0, kd, n) * np.sin(Ad(hs, phi, R, r, theta_p0, n)) * 
                        zeta(theta_p0 + phi + (2 * np.pi) / n, R, r))

# 计算平面之间的夹角
def angleBetweenPlanes(pt1, pt2, pt3, pt4, pt5, pt6):
    n1 = np.cross(pt2 - pt1, pt3 - pt1)
    n2 = np.cross(pt5 - pt4, pt6 - pt4)
    return np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))

# 势能函数
def potentialEnergy(pt1, pt2, pt3, pt4, pt5, pt6, k_, theta0_):
    theta = angleBetweenPlanes(pt1, pt2, pt3, pt4, pt5, pt6)
    return 0.5 * k_ * (theta - theta0_)**2

# 计算 Kresling 结构的磁能量
def KreslingMagnetic(R, r, n, theta_p0, phi_p, hp):
    Kd = 0.8 * 10**6
    M = np.sqrt(2 * Kd / mu0)
    V = (10 / 1000) * np.pi * (5 / 1000)**2
    magm = M * V
    mag = [(-1, 1) if i % 2 == 0 else (1, -1) for i in range(n)]
    
    Points1 = np.array([[R * np.cos(i / n * 2 * np.pi), R * np.sin(i / n * 2 * np.pi), 0] for i in range(n)])
    Points2 = np.array([[r * np.cos(i / n * 2 * np.pi + theta_p0 + phi_p), 
                         r * np.sin(i / n * 2 * np.pi + theta_p0 + phi_p), hp] for i in range(n)])
    Points = np.vstack([Points1, Points2])
    
    magForce = []
    magEnergy = 0
    for i in range(n):
        for j in range(n, 2 * n):
            magForce.append(Fmag([0, 0, mag[i][0] * magm], [0, 0, mag[j][0] * magm], 
                                  rij(i, j, Points), mu0))
            magEnergy += Emag([0, 0, mag[i][0] * magm], [0, 0, mag[j][0] * magm], 
                               rij(i, j, Points), mu0)
    
    return magEnergy

# 弹簧势能和磁能量的总能量函数
def totalU(hpS1, phiS1, R, r, theta_p0, hp0, n):
    kc = 1
    return n / 2 * (kc * ((c(hpS1, phiS1, r, R, theta_p0) - c(hpS1, 0, r, R, theta_p0))**2) + 
                    kc * ((d(hpS1, phiS1, r, R, theta_p0, n) - d(hpS1, phiS1, r, R, theta_p0, n))**2)) + \
           KreslingMagnetic(R, r, n, theta_p0, phiS1, hpS1)[0]  # 这里只是示例，返回磁能量部分



# 定义总力函数 Ftotal（假设已经定义过）
def Ftotal(hs, phi, r, R, theta_p0, hp0, kc, kd, n):
    return n * (Fcc(hs, phi, r, R, theta_p0, hp0, kc) * np.cos(Ac(hs, phi, R, r, theta_p0)) +
                Fdd(hs, phi, r, R, theta_p0, hp0, kd, n) * np.cos(Ad(hs, phi, R, r, theta_p0, n)))

# 约束方程
def g1(vars, R, r, theta_p0, n):
    hpS1, phiS1, hpS2, phiS2 = vars
    # totalU[hpS1, phiS1, 2, 1, Pi/3, 1, 6] + totalU[hpS2, phiS2, 3, 2, Pi/3, 1, 6] 
    return totalU(hpS1, phiS1, R, r, theta_p0, 1, n) + totalU(hpS2, phiS2, R, r, theta_p0, 1, n) - 1  # 假设右边为某个值

def g2(vars, R, r, theta_p0, n):
    hpS1, phiS1, hpS2, phiS2 = vars
    # Ftotal[hpS1, phiS1, 1, 2, Pi/3, 1, 1, 1, 6] + Total[Take[KreslingMagnetic[1, 2, 6, 1, phiS1, hpS1][[2]], 6]][[3]] 
    # - Ftotal[hpS2, phiS2, 2, 3, Pi/3, 1, 1, 1, 6] - Total[Take[KreslingMagnetic[2, 3, 6, 1, phiS2, hpS2][[2]], 6]][[3]] == 0
    return Ftotal(hpS1, phiS1, 1, 2, theta_p0, 1, 1, 1, n) + np.sum(KreslingMagnetic(R, r, n, theta_p0, phiS1, hpS1)[1]) - \
           Ftotal(hpS2, phiS2, 2, 3, theta_p0, 1, 1, 1, n) - np.sum(KreslingMagnetic(R, r, n, theta_p0, phiS2, hpS2)[1])  # == 0

def g3(vars, R, r, theta_p0, n):
    hpS1, phiS1, hpS2, phiS2 = vars
    # Ttotal[hpS1, phiS1, 1, 2, Pi/3, 1, 1, 1, 6] + KreslingMagnetic[1, 2, 6, 1, phiS1, hpS1][[3]] + 
    # Ttotal[hpS2, phiS2, 2, 3, Pi/3, 1, 1, 1, 6] + KreslingMagnetic[2, 3, 6, 1, phiS2, hpS2][[3]] == 0
    return Ttotal(hpS1, phiS1, 1, 2, theta_p0, 1, 1, 1, n) + KreslingMagnetic(R, r, n, theta_p0, phiS1, hpS1)[2] + \
           Ttotal(hpS2, phiS2, 2, 3, theta_p0, 1, 1, 1, n) + KreslingMagnetic(R, r, n, theta_p0, phiS2, hpS2)[2]  # == 0

# 定义方程组
def equations(vars, R, r, theta_p0, n):
    hpS1, phiS1, hpS2, phiS2 = vars
    return [g1(vars, R, r, theta_p0, n), g2(vars, R, r, theta_p0, n), g3(vars, R, r, theta_p0, n)]

# 初始猜测值
initial_guess = [0.5, 0.5, 0.5, 0.5]  # 假设初始猜测值

# 使用 fsolve 求解约束方程
R = 2  # 示例值
r = 1  # 示例值
theta_p0 = np.pi / 3  # 示例值
n = 6  # 示例值

solution = fsolve(equations, initial_guess, args=(R, r, theta_p0, n))
hpS1_sol, phiS1_sol, hpS2_sol, phiS2_sol = solution

print(f"Solution: hpS1 = {hpS1_sol}, phiS1 = {phiS1_sol}, hpS2 = {hpS2_sol}, phiS2 = {phiS2_sol}")

# 计算目标函数 f(x, y, z) 的值
f_value = totalU(hpS1_sol, phiS1_sol, R, r, theta_p0, 1, n) + totalU(hpS2_sol, phiS2_sol, R, r, theta_p0, 1, n)
print(f"f(hpS1, phiS1, hpS2, phiS2) = {f_value}")

# 进行图像绘制（如果需要）
# 假设 hpS1 是自变量，绘制 f(hpS1) 的图像
hpS1_values = np.linspace(0, 1, 100)
f_values = []

for hpS1 in hpS1_values:
    def equations_with_hpS1(vars):
        phiS1, hpS2, phiS2 = vars
        return [g1([hpS1, phiS1, hpS2, phiS2], R, r, theta_p0, n), 
                g2([hpS1, phiS1, hpS2, phiS2], R, r, theta_p0, n), 
                g3([hpS1, phiS1, hpS2, phiS2], R, r, theta_p0, n)]
    
    solution_with_hpS1 = fsolve(equations_with_hpS1, [0.5, 0.5, 0.5])
    phiS1_sol, hpS2_sol, phiS2_sol = solution_with_hpS1
    f_values.append(totalU(hpS1, phiS1_sol, R, r, theta_p0, 1, n) + totalU(hpS2_sol, phiS2_sol, R, r, theta_p0, 1, n))

plt.plot(hpS1_values, f_values)
plt.xlabel('hpS1')
plt.ylabel('f(hpS1, phiS1, hpS2, phiS2)')
plt.title('Total energy under constraints')
plt.show()
