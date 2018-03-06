import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame
from pandas_ml import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from breakdown.utility import balanceData


def outputClassifierResults(data, classifier, title):
    xCompleteData = data.iloc[:, 1:-1].as_matrix()
    yCompleteData = data.loc[:, "genre"].values

    cmActualLabels = []
    cmPredictedLabels = []
    scores = []
    sss = StratifiedKFold(n_splits=10, shuffle=True)

    for train_index, test_index in sss.split(xCompleteData, yCompleteData):
        xTrain, xTest = xCompleteData[train_index], xCompleteData[test_index]
        yTrain, yTest = yCompleteData[train_index], yCompleteData[test_index]
        classifier.fit(xTrain, yTrain)
        score = classifier.score(xTest, yTest)
        scores.append(score)
        print(score)
        cmActualLabels.extend(yTest.tolist())
        cmPredictedLabels.extend(classifier.predict(xTest).tolist())

    averageScore = np.mean(scores)
    with open("../out/" + title.replace(" ", "-") + "-Score.txt", "w") as f:
        f.write(str(scores) + "\n")
        f.write(str(averageScore) + "\n")
    print("Average:", averageScore)

    cm = ConfusionMatrix(cmActualLabels, cmPredictedLabels)
    stats = cm.stats_class.loc[[
        "PPV: Pos Pred Value (Precision)",
        "TPR: (Sensitivity, hit rate, recall)",
        "F1 score"
    ]].astype(float).round(4)
    stats.to_csv("../out/" + title.replace(" ", "-") + "-Stats.csv")
    cm.plot()
    plt.title(title)
    plotOutputPath = "../out/" + title.replace(" ", "-") + "-CM.png"
    plt.savefig(plotOutputPath)
    plt.show()


def main():
    fullData: DataFrame = pandas.read_csv("../data/processed/fullLabeledData.csv", index_col=0)
    fullData = fullData[fullData["genre"] != "New"]
    fullData = fullData[fullData["genre"] != "World"]
    fullData = fullData[fullData["genre"] != "Blues"]
    # The following two lines can be un commented to improve accuracy scores by removing this difficult data :p
    # fullData = fullData[fullData["genre"] != "Electronic"]
    # fullData = fullData[fullData["genre"] != "Jazz"]
    balancedData = balanceData(fullData, targetColumn="genre", n_samples=1500, replace=False)

    nJobs = 6

    sgd = SGDClassifier(max_iter=1000, tol=1e-3, n_jobs=nJobs)
    print("Stochastic Gradient Descent")
    outputClassifierResults(balancedData, sgd, title="Stochastic Gradient Descent")

    rf = RandomForestClassifier(n_jobs=nJobs)
    print("Random Forest")
    outputClassifierResults(balancedData, rf, title="Random Forest")

    ab = AdaBoostClassifier()
    print("Ada Boost")
    outputClassifierResults(balancedData, ab, title="Ada Boost")

    et = ExtraTreesClassifier(n_jobs=nJobs)
    print("Extra Trees")
    outputClassifierResults(balancedData, et, title="Extra Trees")

    gb = GradientBoostingClassifier()
    print("Gradient Boosting")
    outputClassifierResults(balancedData, gb, title="Gradient Boosting")

    gbOneVsRest = OneVsRestClassifier(gb, n_jobs=nJobs)
    print("Gradient Boosting One Vs Rest")
    outputClassifierResults(balancedData, gbOneVsRest, title="Gradient Boosting One Vs Rest")

    gbOneVOne = OneVsOneClassifier(gb, n_jobs=nJobs)
    print("Gradient Boosting One Vs One")
    outputClassifierResults(balancedData, gbOneVOne, title="Gradient Boosting One Vs One")

    b = BaggingClassifier(n_jobs=nJobs)
    print("Bagging Classifier")
    outputClassifierResults(balancedData, b, title="Bagging")

    pa = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3, n_jobs=nJobs)
    print("Passive Aggressive")
    outputClassifierResults(balancedData, pa, title="Passive Aggressive")

    r = RidgeClassifier(max_iter=1000, tol=1e-3)
    print("Ridge Classifier")
    outputClassifierResults(balancedData, r, title="Ridge")

    voting1 = VotingClassifier(estimators=[
        ("sgd", sgd), ("rf", rf), ("ab", ab), ("et", et), ("gb", gb), ("b", b), ("r", r),
        ("gbOvR", gbOneVsRest), ("gbOvO", gbOneVOne)
    ])
    print("Voting (SGD, RF, AB, GB, B, R)")
    outputClassifierResults(balancedData, voting1, title="Voting (SGD, RF, AB, GB, B, R, GBOvR, GBOvO)")

    voting2 = VotingClassifier(estimators=[
        ("rf", rf), ("ab", ab), ("et", et), ("gb", gb), ("b", b), ("r", r), ("gbOvR", gbOneVsRest), ("gbOvO", gbOneVOne)
    ])
    print("Voting (RF, AB, GB, B, R)")
    outputClassifierResults(balancedData, voting2, title="Voting (RF, AB, GB, B, R, GBOvR, GBOvO)")

    # weights made from the average accuracy of the composite models
    voting3 = VotingClassifier(estimators=[
        ("rf", rf), ("ab", ab), ("et", et), ("gb", gb), ("b", b), ("r", r), ("gbOvR", gbOneVsRest), ("gbOvO", gbOneVOne)
    ], weights=[0.3, 0.31, 0.29, 0.34, 0.29, 0.31, 0.35, 0.35])
    print("Weighted Voting (RF, AB, GB, B, R, GBOvR, GBOvO)")
    outputClassifierResults(balancedData, voting3, title="Weighted Voting (RF, AB, GB, B, R, GBOvR, GBOvO)")


if __name__ == '__main__':
    main()
