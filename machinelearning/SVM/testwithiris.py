print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

def make_meshgrid(x, y, h=.02):
    '''
        create a mesh of points to plot
    :param x:data to base x-axis meshgrid on
    :param y:data to base y-axis meshgrid on
    :param h:stepsize for meshgrid, optional
    :return: xx,yy ndarry
    '''
    x_min, x_max = x.min()-1, x.max() + 1
    y_min, y_max = x.min()-1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    '''
        plot the decision boundaries for a classifer
    :param ax:matplotlib axes object
    :param clf:a classifier
    :param xx:meshgrid ndarray
    :param yy:meshgrid ndarray
    :param params:
    :return:
    '''
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour(xx, yy, Z, **params)
    return out

# import some data to play with
iris = datasets.load_iris()
# take the first two features.we could avoid this by using a two-dim dataset
X = iris.data[:,:2]
y = iris.target

C = 1.0
models = (svm.SVC(kernel='linear',C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf',gamma=0.7,C=C),
          svm.SVC(kernel='poly',degree=3,C=C))

models = (clf.fit(X, y) for clf in models)
# title for the plot
titles = ('SVC with linear model',
           'LinearSVC (linear kernel)',
           'SVC with RBF kernel',
           'SVC with polymial (degree 3)kernel')

fig, sub = plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[ : , 0],X[ : , 1]
xx, yy =make_meshgrid(X0,X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax,clf, xx, yy,
                  cmap=plt.cm.coolwarm,alpha=0.8)
    ax.scatter(X0,X1,c=y,cmap=plt.cm.coolwarm,s=20,edgecolor = 'k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()