import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# collect training data
data_set = load_iris()
test_idx = [0, 50, 100]

def main():
    print('iris.feature_names={}'.format(data_set.feature_names))
    print('iris.target_names={}'.format(data_set.target_names))
    print('iris sample data={}'.format(data_set.data[0]))
    # print (iris.target[0])
    # for i in range(len(iris.target)):
    # 	print ("Example {}: label {} features {}".format(i, iris.target[i], iris.data[i]))

    # training data
    train_target = np.delete(data_set.target, test_idx)
    train_data = np.delete(data_set.data, test_idx, axis=0)

    # testing data
    test_target = data_set.target[test_idx]
    test_data = data_set.data[test_idx]

    # training the classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    # make predictions using the trained classifier
    print('test_target={}'.format(test_target))
    print('clf prediction={}'.format(clf.predict(test_data)))

    # visualizing the tree
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=data_set.feature_names,
                                    class_names=data_set.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./iris.pdf")
