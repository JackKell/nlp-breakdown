from typing import List

import pandas
from pandas import DataFrame
import numpy
from numpy import ndarray
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC


def main():
    fullData: DataFrame = pandas.read_csv("../data/processed/fullLabeledData.csv", index_col=0)
    # print(fullData)

    xData = fullData.iloc[:, 1:-1].as_matrix()
    yData = fullData.loc[:, "genre"].values

    # print(xData.shape)
    # print(yData.shape)

    scores: List = []
    sss: StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=10, test_size=0.9, random_state=0)
    for train_index, test_index in sss.split(xData, yData):
        xTrain = xData[train_index]
        xTest = xData[test_index]
        yTrain, yTest = yData[train_index], yData[test_index]

        clf: SVC = SVC(cache_size=5000).fit(xTrain, yTrain)
        score = clf.score(xTest, yTest)
        print(score)
        scores.append(score)

    print(scores)
    print(numpy.mean(scores))


if __name__ == '__main__':
    main()
