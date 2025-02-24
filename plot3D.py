# %%
import numpy as np
import plotly.graph_objects as go
# 提取数据
x = results[:, 1]  # 第二列
y = results[:, 2]  # 第三列
z = results[:, 3]  # 第四列
t = results[:, 0]  # 第一列作为变化方向

# 计算切向量
dx = np.diff(x)
dy = np.diff(y)
dz = np.diff(z)

# 选择箭头的间隔，假设我们每隔10个点添加一个箭头
arrow_interval = 10

# 创建箭头数据
arrow_x = x[:-1:arrow_interval]
arrow_y = y[:-1:arrow_interval]
arrow_z = z[:-1:arrow_interval]
arrow_dx = dx[::arrow_interval]
arrow_dy = dy[::arrow_interval]
arrow_dz = dz[::arrow_interval]

# 归一化方向向量
magnitude = np.sqrt(arrow_dx**2 + arrow_dy**2 + arrow_dz**2)
arrow_dx /= magnitude
arrow_dy /= magnitude
arrow_dz /= magnitude


# 创建3D图形
fig = go.Figure()

# 绘制三维轨迹
fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Trajectory', line=dict(color='blue')))

fig.add_trace(go.Cone(x=arrow_x, y=arrow_y, z=arrow_z, u=arrow_dx, v=arrow_dy, w=arrow_dz, 
                     colorscale='Viridis', showscale=False, sizemode="absolute", sizeref=1))


x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
z_min, z_max = np.min(z), np.max(z)

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        zaxis=dict(range=[z_min, z_max]),
    ),
)


# 显示图像
fig.show()
