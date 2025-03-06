import matplotlib.pyplot as plt
from math import exp
import numpy as np

x = np.random.randn(256)
x.sort()

Sigmoid = lambda x: exp(x)/(exp(x)+1)
softmax = lambda x: exp(x)

y = [Sigmoid(_) for _ in x]
y_softmax = [softmax(_) for _ in x]

y = list(map(lambda x: x/sum(y), y))
y_softmax = list(map(lambda x: x/sum(y_softmax), y_softmax))
print(y)
print(y_softmax)

# 计算累积和
cumulative_sum_y = np.cumsum(y)
cumulative_sum_y_softmax = np.cumsum(y_softmax)
print(cumulative_sum_y)
print(cumulative_sum_y_softmax)

# 创建主图和两个子图
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 7))  # 调整figsize以适应两个子图

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_yscale('log')

ax1.plot(x, cumulative_sum_y, label='Cumulative Sum of Sigmoid', color='green', linestyle='-')
ax1.plot(x, cumulative_sum_y_softmax, label='Cumulative Sum of softmax', color='blue', linestyle='-')
ax2.plot(x, y, label='Sigmoid', color='blue', linestyle='--')
# 绘制正态分布的概率密度函数，采样点数为256
from scipy.stats import norm
mu, sigma = 0, 1
x_norm = np.linspace(min(x), max(x), 256)
y_norm = norm.pdf(x_norm, mu, sigma)
ax2.plot(x_norm, y_norm, label='Normal Distribution', color='purple', linestyle='-')

# Add softmax with temperature
T_values = [0.5, 1.0, 2.0, 5.0]  # Example temperature values
for T in T_values:
    softmax_with_temp = lambda x: exp(x/T)
    y_softmax_with_temp = [softmax_with_temp(_) for _ in x]
    y_softmax_with_temp = list(map(lambda x: x/sum(y_softmax_with_temp), y_softmax_with_temp))
    # Adjust color intensity based on temperature
    # color_intensity = T / max(T_values)  # Normalize T to be between 0 and 1
    # color = (0, 0, 1 - (1-color_intensity)*0.7)  # Blue color, darker as T increases
    # Use a yellow-to-red colormap
    color = plt.cm.YlOrRd((T - min(T_values)+0.1) / (max(T_values) - min(T_values)+0.1))
    ax2.plot(x, y_softmax_with_temp, label=f'softmax with T={T}', color=color, linestyle='--')


ax1.set_xlabel('x')
ax1.set_ylabel('Cumulative Sum', color='black')
ax2.set_ylabel('Normalized Value', color='black')

ax1.set_title('Comparison of Sigmoid and softmax')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine the handles and labels
handles = handles1 + handles2
labels = labels1 + labels2

# Create a single legend
ax1.legend(handles, labels, loc='upper left')

# Third subplot for comparing softmax with high temperature to Sigmoid
for T in T_values:
    softmax_with_temp = lambda x: exp(x/T)
    y_softmax_with_temp = [softmax_with_temp(_) for _ in x]
    y_softmax_with_temp = list(map(lambda x: x/sum(y_softmax_with_temp), y_softmax_with_temp))
    color = plt.cm.YlOrRd((T - min(T_values) + 0.1) / (max(T_values) - min(T_values) + 0.1))
    ax3.plot(x, y_softmax_with_temp, label=f'softmax with T={T}', color=color, linestyle='--')

ax3.plot(x, y, label='Sigmoid', color='blue', linestyle='--')  # Plot Sigmoid for comparison
ax3.plot(x, 1/len(x)*np.ones(len(x)), label='Avearge Value', color='red', linestyle='-')  # Plot average value for comparison

ax3.set_xlabel('x')
ax3.set_ylabel('Normalized Value')
ax3.set_ylim(0, 1.2*max(y))  # Set x-axis limits to match the first subplot
ax3.set_title('Detailed Comparison of Sigmoid and softmax with High Temperature')
ax3.legend()


# plt.grid()
plt.tight_layout()
plt.show()
