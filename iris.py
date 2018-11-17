import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


def main():
    # collect training data
    iris = load_iris()
    test_idx = [0, 50, 100]

    print('iris.feature_names={}'.format(iris.feature_names))
    print('iris.target_names={}'.format(iris.target_names))
    print('iris sample data={}'.format(iris.data[0]))
    # print (iris.target[0])
    # for i in range(len(iris.target)):
    # 	print ("Example {}: label {} features {}".format(i, iris.target[i], iris.data[i]))

    # training data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    # testing data
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    # training the classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    # make predictions using the trained classifier
    print('test_target={}'.format(test_target))
    print('clf prediction={}'.format(clf.predict(test_data)))

    # visualizing the tree
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("./iris.pdf")


# execute main function - entry point
main()
