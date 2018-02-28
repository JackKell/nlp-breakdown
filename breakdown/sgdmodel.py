from typing import List

import pandas
from pandas import DataFrame
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile
from time import time
import matplotlib.pyplot as plt

from breakdown.utility import plot_confusion_matrix


def main():
    fullData: DataFrame = pandas.read_csv("../data/processed/fullLabeledData.csv", index_col=0)
    fullData = fullData[fullData["genre"] != "Rock"]
    fullData = fullData[fullData["genre"] != "Pop"]
    # fullData = fullData[fullData["genre"] != "Metal"]
    fullData = fullData[fullData["genre"] != "New"]

    xCompleteData = fullData.iloc[:, 1:-1].as_matrix()
    print(xCompleteData.shape)
    print(fullData["genre"].value_counts())
    yCompleteData = fullData.loc[:, "genre"].values
    classNames = set(yCompleteData)
    # xCompleteData = SelectPercentile(chi2, percentile=10).fit_transform(xCompleteData, yCompleteData)
    print(xCompleteData.shape)

    # sss: StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    # trainingIndexes, holdoutIndexes = next(sss.split(xCompleteData, yCompleteData))
    # xCompleteTrain, xHoldout = xCompleteData[trainingIndexes], xCompleteData[holdoutIndexes]
    # yCompleteTrain, yHoldout = yCompleteData[trainingIndexes], yCompleteData[holdoutIndexes]

    xCompleteTrain, xHoldout, yCompleteTrain, yHoldout = train_test_split(xCompleteData, yCompleteData, random_state=0)

    scores: List = []
    start = time()
    foldedCF = None

    cmT = []
    cmP = []
    sss: StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    clf = SGDClassifier(max_iter=1000, tol=1e-3)
    for train_index, test_index in sss.split(xCompleteTrain, yCompleteTrain):
        xTrain, xTest = xCompleteData[train_index], xCompleteData[test_index]
        yTrain, yTest = yCompleteData[train_index], yCompleteData[test_index]
        clf.fit(xTrain, yTrain)
        score = clf.score(xTest, yTest)
        print(score)
        scores.append(score)
        # currentCF = confusion_matrix(yTest, clf.predict(xTest))
        cmT.extend(yTest)
        cmP.extend(clf.predict(xTest))
    end = time()


    print(scores)
    print(numpy.mean(scores))
    print(end - start)
    clf.fit(xCompleteTrain, yCompleteTrain)
    print(clf.score(xHoldout, yHoldout))
    yHoldoutPredictions = clf.predict(xHoldout)

    # Compute confusion matrix
    # cnf_matrix = confusion_matrix(yHoldout, yHoldoutPredictions)
    cnf_matrix = confusion_matrix(cmT, cmP)
    numpy.set_printoptions(precision=1)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classNames, title='Confusion matrix, without normalization')
    print(cnf_matrix)
    plt.show()

    # # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=yHoldout, normalize=True, title='Normalized confusion matrix')
    #
    # plt.show()

    # print(list(yHoldout))
    # print(list(yHoldoutPredictions))


if __name__ == '__main__':
    main()
