from demo.PerceptionDemo import Perceptron

# 定义激活函数f
f = lambda x: x


def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = Perceptron(1, f)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元
    return lu


import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy import *


def draw():
    # 画图
    dataMat, labelMat = get_training_dataset()
    dataArr = array(dataMat)
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    n = shape(dataArr)[0]  # number of points to create
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i])
            ycord1.append(labelMat[i])
        else:
            xcord2.append(dataArr[i])
            ycord2.append(labelMat[i])
    fig = plt.figure()
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax = fig.add_subplot(111)
    type1 = ax.scatter(xcord1, ycord1, s=10, c='red', marker='s')
    type2 = ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(0, 12.0, 0.1)
    line, = ax.plot(x, x)

    def data_gen():
        i = 0
        y = x
        p = Perceptron(1, f)
        while i < 10:
            yield y
            p._one_iteration(dataArr, labelMat, 0.01)
            y = (p.bias + p.weights[0] * x)
            i += 1
            print(p)

    def update(data):
        line.set_ydata(data)
        return line,

    ani = animation.FuncAnimation(fig, update, data_gen(), interval=1000)
    plt.show()


if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    draw()
