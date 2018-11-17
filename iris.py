import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


def main():
    iris = load_iris()
    test_idx = [0, 50, 100]

    # print (iris.feature_names)
    # print (iris.target_names)
    # print (iris.data[0])
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

    print(test_target)


# execute main function - entry point
main()
