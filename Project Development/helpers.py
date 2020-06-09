import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets, tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from ipywidgets import interactive

np.random.seed(189)

#From http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
X1 = X + 0.3*np.random.randn(X.shape[0], X.shape[1])
y = iris.target
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

n_neighbors = 1
weights ='uniform'


#NOTE: The following section is for the visualization of decision trees

#From http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

# Load data
def plotPairwiseDecisionTrees(tree_depth=None):
    plt.figure(figsize = (10,7))
    # Parameters
    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X1 = iris.data[:, pair]
        y = iris.target

        # Train
        clf = DecisionTreeClassifier(max_depth=tree_depth).fit(X1, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
        y_min, y_max = X1[:, 1].min() - 1, X1[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X1[idx, 0], X1[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    if tree_depth == None:
        plt.suptitle("Decision surface of a decision tree using paired features with max depth")
    else:
        plt.suptitle("Decision surface of a decision tree using paired features with depth %i" %tree_depth )
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()


#NOTE: This is for the interactive visualization of decision trees.

#Source: https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084

def plot_tree(data, crit='entropy', split='best', min_samples_split=2, class_weight=None, max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_weight_fraction_leaf=0.0, presort=False):

    def plot_tree_helper(crit, split, depth, min_samples_split):
        estimator = DecisionTreeClassifier(
              random_state = 0
            , criterion = crit
            , splitter = split
            , min_samples_split = min_samples_split
            , class_weight = class_weight
            , max_depth = max_depth
            , max_features = max_features
            , max_leaf_nodes = max_leaf_nodes
            , min_impurity_decrease = min_impurity_decrease
            , min_impurity_split = min_impurity_split
            , min_samples_leaf = min_samples_leaf
            , min_weight_fraction_leaf = min_weight_fraction_leaf
            , presort = presort)
        estimator.fit(data.data, data.target)
        graph = Source(tree.export_graphviz(estimator
                , out_file = None
                , feature_names=data.feature_names
                , class_names=[str(name) for name in np.unique(data.target)]
                , filled = True))
        display(SVG(graph.pipe(format='svg')))
        return graph

    estimator = DecisionTreeClassifier(
              random_state = 0
            , criterion = crit
            , splitter = split
            , min_samples_split = min_samples_split
            , class_weight = class_weight
            , max_depth = max_depth
            , max_features = max_features
            , max_leaf_nodes = max_leaf_nodes
            , min_impurity_decrease = min_impurity_decrease
            , min_impurity_split = min_impurity_split
            , min_samples_leaf = min_samples_leaf
            , min_weight_fraction_leaf = min_weight_fraction_leaf
            , presort = presort)
    estimator.fit(data.data, data.target)
    total_depth = estimator.tree_.max_depth

    inter = interactive(plot_tree_helper, crit=crit, split=split, min_samples_split=min_samples_split, class_weight = class_weight, max_depth = max_depth, max_features = max_features, max_leaf_nodes = max_leaf_nodes, min_impurity_decrease = min_impurity_decrease, min_impurity_split = min_impurity_split, min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf, presort = presort, random_state = 0, depth=list(range(1, total_depth + 1)))
    display(inter)
