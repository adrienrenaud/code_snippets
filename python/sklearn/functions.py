import pydotplus
from IPython.display import Image
from sklearn import tree


# PLoting


def plot_decision_tree(model, features):
    """Plot a Decision tree
    Args:
        - model (sklearn.tree.DecisionTreeClassifier): the dt to plot
        - features (list): the list of features names to be used (need to be inn the same order as for training)
    """
    dot_data = tree.export_graphviz(model, out_file=None, 
                     feature_names=features,  
                     class_names=['0', '1'] ,  
                     filled=True, rounded=True, rotate=True, 
                     special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)  
    return Image(graph.create_png())
