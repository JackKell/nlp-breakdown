from typing import List

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer

from breakdown.utility import top_mean_feats
import pandas


class TrackDataItem(object):
    def __init__(self, trackId: str, mxmTrackId: str, wordCountList: List[str], label: str):
        self.trackId: str = trackId
        self.mxmTrackId: str = mxmTrackId
        self.wordCountList: List[str] = wordCountList
        self.label = label

    def getWordCountListAsString(self, wordsList) -> str:
        lyricsWordList = []
        for wordCount in self.wordCountList:
            wordIndex, count = wordCount.split(":")
            extra = [wordsList[int(wordIndex) - 1]] * int(count)
            lyricsWordList.extend(extra)
        return " ".join(lyricsWordList)


def main():
    lyricDataPath = "../data/raw/mxm_dataset_train.txt"
    trackLablesPath = "../data/raw/trackid_label.txt"

    trackLabelMap = {}
    labels = set()
    with open(trackLablesPath) as f:
        lines = f.readlines()
        for line in lines:
            trackId, label = line.split(",")[:2]
            trackLabelMap[trackId] = label
            labels.add(label)

    with open(lyricDataPath, "r") as f:
        fileLines = f.readlines()
        wordsList = fileLines[17][1:].split(",")
        trackDataLines = fileLines[18:][1:]

    trackDataItems = []
    for trackDataLine in trackDataLines:
        trackFeatures = trackDataLine.replace("\n", "").split(",")
        trackId = trackFeatures[0]
        mxmId = trackFeatures[1]
        wordCountList = trackFeatures[2:]
        trackDataItems.append(TrackDataItem(trackId, mxmId, wordCountList, trackLabelMap.get(trackId)))

    lyricsList = [trackDataItem.getWordCountListAsString(wordsList) for trackDataItem in trackDataItems]

    globalLyricVectorizer = TfidfVectorizer(norm="l2", stop_words="english")
    sparseTFIDFMatrix = globalLyricVectorizer.fit_transform(lyricsList)
    # print(globalLyricVectorizer.vocabulary_)

    denseTFIDFMatrix = sparseTFIDFMatrix.toarray()

    topN = 100
    # print("Top", topN, "most important words")
    # print(top_mean_feats(sparseTFIDFMatrix, globalLyricVectorizer.get_feature_names(), top_n=topN))

    topFeatureLists = []
    for label in labels:
        labledLyricsList = [trackDataItem.getWordCountListAsString(wordsList) for trackDataItem in trackDataItems if
                            trackDataItem.label == label]
        currentVectorizer = TfidfVectorizer(norm="l2", max_features=topN, stop_words="english")
        sparseTFIDFMatrix = currentVectorizer.fit_transform(labledLyricsList)
        # print("Top", topN, "most important words for", label)
        # print(top_mean_feats(sparseTFIDFMatrix, currentVectorizer.get_feature_names(), top_n=topN))
        # for feature in currentVectorizer.get_feature_names():
        #     importantFeatures.add(feature)
        topFeatureLists.append(currentVectorizer.get_feature_names())

    for i in range(5, 55, 5):
        importantFeatures = set()
        for topFeatureList in topFeatureLists:
            for feature in topFeatureList[:i]:
                importantFeatures.add(feature)

        importantFeatures = list(importantFeatures)
        importantFeatureIndexes = [
            globalLyricVectorizer.vocabulary_.get(importantFeature) for importantFeature in importantFeatures
        ]
        # print(importantFeatures)
        # print(importantFeatureIndexes)
        totalFeatures = len(importantFeatures)
        print("Top:", i, "Total Features:", totalFeatures)

        tracksDF = DataFrame(
            [trackDataItem.trackId for trackDataItem in trackDataItems],
            columns=["trackid"]
        )

        featuresDF = DataFrame(
            denseTFIDFMatrix[:, importantFeatureIndexes],
            columns=importantFeatures
        )

        labelDF = DataFrame(
            [trackDataItem.label for trackDataItem in trackDataItems],
            columns=["genre"]
        )

        labeledDataDF: DataFrame = pandas.concat([tracksDF, featuresDF, labelDF], axis=1)
        labeledDataDF = labeledDataDF.dropna(axis=0)
        labeledDataDF.to_csv("../data/processed/fullLabeledDataTop" + str(i) + ".csv")


if __name__ == '__main__':
    main()
