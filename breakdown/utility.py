import numpy
import pandas
import sklearn.utils


def balanceData(unbalancedData, targetColumn, n_samples, replace=False):
    targetLabels = unbalancedData[targetColumn]
    labels = set(list(targetLabels.values))
    balancedData = pandas.DataFrame(columns=unbalancedData.columns.values)
    for label in labels:
        labelData = unbalancedData[unbalancedData[targetColumn] == label]
        sampleDF = sklearn.utils.resample(labelData, n_samples=n_samples, replace=replace)
        balancedData = pandas.concat([balancedData, sampleDF], ignore_index=True)
    return balancedData


# following function credited to https://buhrmann.github.io/tfidf-analysis.html
def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = numpy.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pandas.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = numpy.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = numpy.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
