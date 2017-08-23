# 导入必要的模块
import matplotlib.pyplot as plt

# 产生测试数据
x = [0, 1, 0, 1]
y = [1, 1, 0, 0]
fig = plt.figure()
ax1 = fig.add_subplot(111)
# 设置标题
# ax1.set_title('Scatter Plot')
# 设置X轴标签
# plt.xlabel('X')
# 设置Y轴标签
# plt.ylabel('Y')
# 画散点图
ax1.scatter(x, y, c='r', marker='o')
# 设置图标
plt.legend('x1')
# 显示所画的图
# plt.show()

x = [1, 2, 3]

y_1 = [4, 5, 6]
y_2 = [2, 3, 4]

plt.plot(x, y_1, marker='x')
plt.plot(x, y_2, marker='^')

plt.xlim([-1, 9])
plt.ylim([-1, 9])
plt.xlabel('x-axis label')
plt.ylabel('y-axis label')
plt.title('Simple line plot')
plt.legend(['sample 1', 'sample2'], loc='upper left')

plt.show()
