from sklearn.svm import LinearSVC, SVC
import mglearn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


#
# fig, axes = plt.subplots(3,3,figsize=(15,10))
#
# for ax, c in zip(axes, [-1, 0, 3]):
#     for a, gamma in zip(ax, range(-1,2)):
#         mglearn.plots.plot_svm(log_C=c, log_gamma=gamma, ax = a)
#
# axes[0,0].legend(['class 0', 'class 1', 'class 0 support vector', 'class 1support vector'],
#                  ncol=4, loc=(.9,1.2))
# plt.show()



x,y = make_moons(n_samples=100, noise= 0.25, random_state=3)

x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, random_state=42)

fig, axes = plt.subplots(2,4,figsize=(15,10))

for axx, n_hidden_node in zip(axes, [10,20]):
    for ax, alphaVal in zip(axx, [0.0001, 0.01, 0.1, 1]):
        mlp = MLPClassifier(solver='lbfgs', random_state=0, max_iter= 1000,
        hidden_layer_sizes=[n_hidden_node, n_hidden_node],alpha= alphaVal)
        mlp.fit(x_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, x_train, fill=True, alpha= .3, ax=ax)
        mglearn.discrete_scatter(x_train[:,0], x_train[:,1], y_train, ax=ax)
        ax.set_title("n_hidden=[{},{}], \n alpha={:.4f}".format(n_hidden_node, n_hidden_node, alphaVal))

plt.show()