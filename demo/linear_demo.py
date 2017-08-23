import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def runplt():
    plt.figure()
    plt.title('price  and inch')
    plt.xlabel('inch')
    plt.ylabel('price')
    plt.axis([0, 25, 0, 25])
    plt.grid(True)
    return plt


X = [[6], [8], [10], [14], [18]]
y = [[7], [9], [13], [17.5], [18]]
plt = runplt()
model = LinearRegression()
model.fit(X, y)
X2 = [[3], [10], [14], [20]]
y2 = model.predict(X2)
plt.plot(X, y, 'k.-')
plt.plot(X2, y2, 'g.-')
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')
plt.show()
