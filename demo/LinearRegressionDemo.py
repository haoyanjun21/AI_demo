'''
Created on 2017-08-23
@author: hyj
'''
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from numpy import *


def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('LinearRegressionData.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def stocGradAscentFixStep(dataMatrix, classLabels, weights):
    m, n = shape(dataMatrix)
    alpha = 0.01
    for i in range(m):
        y = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - y
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscentVaryStep(dataMatrix, classLabels, weights):
    m, n = shape(dataMatrix)
    dataIndex = list(range(m))
    for i in dataIndex:
        alpha = 4 / (1.0 + i) + 0.0001  # apha decreases with iteration, does not
        randIndex = int(random.uniform(0, len(dataIndex)))  # go to 0 because of the constant
        y = sigmoid(sum(dataMatrix[randIndex] * weights))
        error = classLabels[randIndex] - y
        weights = weights + alpha * error * dataMatrix[randIndex]
        del dataIndex[randIndex]
    return weights


dataMat, labelMat = loadDataSet()
dataArr = array(dataMat)

n = shape(dataArr)[0]  # number of points to create


def xycord(dataArr, labelMat):
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    return xcord1, ycord1, xcord2, ycord2


xcord1, ycord1, xcord2, ycord2 = xycord(dataArr, labelMat)

fig = plt.figure()
plt.xlabel('X1')
plt.ylabel('X2')
ax = fig.add_subplot(111)
type1 = ax.scatter(xcord1, ycord1, s=10, c='red', marker='s')
type2 = ax.scatter(xcord2, ycord2, s=30, c='green')
x = arange(-3.0, 3.0, 0.1)
line, = ax.plot(x, x)


def update(data):
    line.set_ydata(data)
    return line,


def data_gen():
    i = 0
    y = x
    weights = ones(3)
    while i < 160:
        yield y
        print(weights)
        weights = stocGradAscentFixStep(dataArr, labelMat, weights)
        y = (-weights[0] - weights[1] * x) / weights[2]
        i += 1
        line.set_ydata(y)


ani = animation.FuncAnimation(fig, update, data_gen(), interval=100)
plt.show()
