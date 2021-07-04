#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import numpy as np
    import pandas as pd
    import sys
    import argparse
    import string
    import re
    import unicodedata

    import nltk
    import spacy

    # nltk.download('rslp')
    # nltk.download('stopwords')
    # nltk.download('mac_morpho')
    # nltk.download('floresta')

    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn.svm import LinearSVC, SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB, GaussianNB

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer

    from sklearn import metrics
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import balanced_accuracy_score

    from sklearn.utils import resample

    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler 
    from imblearn.combine import SMOTEENN 

    from scipy.sparse import csr_matrix

    from gensim.models import Word2Vec

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

except Exception as e:
    print('Unmet dependency:', e)
    sys.exit(1)

class MeanEmbeddingVectorizer():

    def __init__(self, size=50, min_count=1):
        self.size = size
        self.min_count = min_count

    def fit(self, X):
        w2v = Word2Vec(X, vector_size=self.size, min_count=self.min_count)
        self.word2vec = dict(zip(w2v.wv.index_to_key, w2v.wv.vectors))
        self.dim = len(list(self.word2vec.values())[0])
        return self

    def transform(self, X):
        return csr_matrix([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class Main():

    def __init__(self, dataset_train, dataset_test):
        self.datasetTrain = dataset_train
        self.datasetTest = dataset_test

        self.testSize = 0.5
        self.randomState = 42
        self.kNeighbors = 3
        self.nEstimators = 100
        self.runs = 10  # number of runs

    def resa(self, df):
        try:
            df_majority = df[df.prediction==1]
            df_minority = df[df.prediction==0]

            df_majority_downsampled = resample(df_majority, replace=False, n_samples=2476)

            df_downsampled = pd.concat([df_majority_downsampled, df_minority])

            df_downsampled.prediction.value_counts()

            return df_downsampled

        except Exception as a:
            print('resa', a)

    def readData(self, file1, file2):
        try:
            df1 = pd.read_csv(file1, header=0, delimiter="\t")

            dfX1 = (df1.iloc[:, 0])
            dfy1 = (df1.iloc[:, 1:])
            
            rus = RandomUnderSampler(random_state=42, sampling_strategy='all')
            dfX1, dfy1 = rus.fit_resample(dfX1.values.reshape(-1,1), dfy1)
            
            df2 = pd.read_csv(file2, header=0, delimiter="\t")
            dfy2 = df2.iloc[:, 0]
            dfX2 = list(df2.iloc[:, 1])

            fucReshape = []
            for each in dfX1:
                fucReshape.append((str(each[0])))

            print(dfy1['prediction'].value_counts())
            return fucReshape, list(dfy1['prediction']), dfX2, dfy2

        except Exception as a:
            print("readData", a)

    def convertData(self, X):
        try:
            newX = []

            # nlp = spacy.load("pt_core_news_sm")
            # nlp = spacy.load("pt_core_news_md")
            nlp = spacy.load("pt_core_news_lg")

            lemmatizer = nlp.get_pipe("lemmatizer")

            for eachText in X:

                aux0 = unicodedata.normalize('NFKD', eachText).encode(
                    'ascii', 'ignore').decode('utf-8', 'ignore')

                aux = aux0.split(" ")

                newT = ([each for each in aux if '@' not in each])

                newT = ''.join(c for c in ' '.join(newT) if c not in string.punctuation)

                pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
                newT = re.sub(pat, ' ', newT)

                newT = newT.lower()

                newT = re.sub(r'\d+', ' ', newT)

                newT = re.sub('\s+', ' ', newT).strip()

                # steammer = nltk.stem.RSLPStemmer()

                # newR = ' '.join([token.lemma_ for token in nlp(newT)])

                newX.append(newT)
                # newX.append(newR)

            return newX

        except Exception as a:
            print("convertData", a)

    def fitFuncCV(self, inputData):
        try:
            self.extractor = CountVectorizer(input=inputData,
                                             encoding='utf-8',
                                             # decode_error='strict',
                                             # strip_accents=None,
                                             lowercase=True,
                                             # preprocessor=None,
                                             # tokenizer=None,
                                             stop_words=nltk.corpus.stopwords.words(
                                                 'portuguese'),
                                             # token_pattern='(?u)\b\w\w+\b',
                                             # ngram_range=(1, 1),
                                             # analyzer='word',
                                             # max_df=1.0, min_df=1,
                                             max_features=200000,
                                             # vocabulary=None,
                                             binary=False)

            self.extractor.fit(inputData)
        except Exception as a:
            print('fitFuncCV', a)

    def fitFuncTfidf(self, inputData):
        try:
            self.extractor = TfidfVectorizer(inputData,
                                             encoding='utf-8',
                                             # decode_error='strict',
                                             # strip_accents=None,
                                             lowercase=True,
                                             # preprocessor=None,
                                             # tokenizer=None,
                                             stop_words=nltk.corpus.stopwords.words(
                                                 'portuguese'),
                                             # token_pattern='(?u)\b\w\w+\b',
                                             # ngram_range=(1, 1),
                                             # analyzer='word',
                                             # max_df=1.0, min_df=1,
                                             max_features=40000,
                                             # vocabulary=None,
                                             binary=False)

            self.extractor.fit(inputData)
        except Exception as a:
            print('fitFuncTfidf', a)

    def fitFuncHV(self, inputData):
        try:
            self.extractor = HashingVectorizer(inputData,
                                               n_features=50000,
                                               encoding='utf-8',
                                               # decode_error='strict',
                                               # strip_accents=None,
                                               lowercase=True,
                                               # preprocessor=None,
                                               # tokenizer=None,
                                               stop_words=nltk.corpus.stopwords.words(
                                                   'portuguese'),
                                               # token_pattern='(?u)\b\w\w+\b',
                                               # ngram_range=(1, 1),
                                               # analyzer='word',
                                               # max_df=1.0, min_df=1,
                                               # max_features=None,
                                               # vocabulary=None,
                                               binary=False)

            self.extractor.fit(inputData)
        except Exception as a:
            print('fitFuncHV', a)

    def fitFuncW2V(self, inputData):
        self.extractor = MeanEmbeddingVectorizer(size=1000)

        self.extractor.fit(inputData)

    def transformFunc(self, inputData):
        return self.extractor.transform(inputData)

    def randomForest(self, X_train, X_test, y_train, y_test, THRESHOLD=0.9):
        try:
            clf = RandomForestClassifier(n_estimators=self.nEstimators)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred = [ int(pred[1] > THRESHOLD) for pred in clf.predict_proba(X_test) ]
            return self.printData(y_test, y_pred, 'Random Forest', 'Multi-Class')
        except Exception as a:
            print('randomForest', a)

    def multilayerPerceptron(self, X_train, X_test, y_train, y_test):
        try:
            clf = MLPClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'Multilayer Perceptron', 'Multi-Class')
        except Exception as a:
            print('multilayerPerceptron', a)

    def sgd(self, X_train, X_test, y_train, y_test):
        try:
            clf = SGDClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'SGD', 'Multi-Class')
        except Exception as a:
            print('sgd', a)

    def linearSVC(self, X_train, X_test, y_train, y_test):
        try:
            clf = LinearSVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'linearSVC', 'Multi-Class')
        except Exception as a:
            print('linearSVC', a)

    def svc(self, X_train, X_test, y_train, y_test):
        try:
            clf = SVC()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'SVC', 'Multi-Class')
        except Exception as a:
            print('svc', a)

    def decisionTree(self, X_train, X_test, y_train, y_test):
        try:
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'DecisionTreeClassifier', 'Multi-Class')
        except Exception as a:
            print('decisionTree', a)

    def perceptron(self, X_train, X_test, y_train, y_test):
        try:
            clf = Perceptron()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'Perceptron', 'Multi-Class')
        except Exception as a:
            print('perceptron', a)

    def knn(self, X_train, X_test, y_train, y_test):
        try:
            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'KNeighborsClassifier', 'Multi-Class')
        except Exception as a:
            print('knn', a)

    def multinomialNB(self, X_train, X_test, y_train, y_test):
        try:
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'MultinomialNB', 'Multi-Class')
        except Exception as a:
            print('multinomialNB', a)
            return 0, 0, 0, 0, 0

    def gaussianNB(self, X_train, X_test, y_train, y_test):
        try:
            clf = GaussianNB()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            return self.printData(y_test, y_pred, 'GaussianNB', 'Multi-Class')
        except Exception as a:
            print('gaussianNB', a)

    def printData(self, y_true, y_pred, alg, type):
        try:
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1s = f1_score(y_true, y_pred)
            bal = balanced_accuracy_score(y_true, y_pred)
            # Accuracy, Precision, Recall, F1, and Bacc
            print("Algorithm:\t\t", alg, ' (', type, ')')
            print("Accuracy:", acc)
            print("Precision", pre)
            print("Recall", rec)
            print("F1", f1s)
            print("Bacc", bal)
            return acc, pre, rec, f1s, bal
        except Exception as a:
            print('printData', a)
            return 0, 0, 0, 0, 0

    def featuresLabels(self, features, labels):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=self.testSize)  # , random_state=self.randomState)
            # return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()  # NEWS
            return X_train, X_test, y_train, y_test # TWEETS
        except Exception as a:
            print('featuresLabels', a)

    def main(self):
        # self.makerMain()
        # self.finalNews()
        self.finalTweet()

    def finalTweet(self, THRESHOLD=0.9):
        print('read data')
        x_train, y_train, x_test, y_test = self.readData(self.datasetTrain, self.datasetTest)

        x_train = self.convertData(x_train)
        x_test = self.convertData(x_test)

        self.fitFuncHV(x_train)
        # self.fitFuncTfidf(x_train)
        x_train = self.transformFunc(x_train)

        print('ml time')
        clf = RandomForestClassifier(n_estimators=self.nEstimators)
        clf.fit(x_train, y_train)

        x_test = self.transformFunc(x_test)
        y_pred = clf.predict(x_test)
        y_pred = [ int(pred[1] > THRESHOLD) for pred in clf.predict_proba(x_test) ]

        # print(clf.predict_proba(x_test))
        
        for each in zip(list(y_test), y_pred):
            print(list(each)[0],"\t" ,list(each)[1])

    def finalNews(self, THRESHOLD=0.9):
        print('read data')
        x_train, y_train, x_test, y_test = self.readData(self.datasetTrain, self.datasetTest)

        x_train = self.convertData(x_train)
        x_test = self.convertData(x_test)
       
        self.fitFuncTfidf(x_train)
        x_train = self.transformFunc(x_train)
    
        print('ml time')
        clf = MLPClassifier()
        clf.fit(x_train, y_train)

        x_test = self.transformFunc(x_test)
        y_pred = clf.predict(x_test)
        y_pred = [ int(pred[1] > THRESHOLD) for pred in clf.predict_proba(x_test) ]

        for each in zip(list(y_test), y_pred):
            print(list(each)[0],"\t" ,list(each)[1])

    def makerMain(self):
        # Read dataset
        dfX1, dfy1, dfX2, dfy2 = self.readData(
            self.datasetTrain, self.datasetTest)

        # Convert string arrays
        dfX1 = self.convertData(dfX1)

        results_rf = []
        results_ml = []
        results_sg = []
        results_ls = []
        results_sv = []
        results_dt = []
        results_pe = []
        results_kn = []
        results_mn = []
        results_gn = []

        # train and test run times
        for i in range(self.runs):
            print("=== run {} ===".format(i + 1))
            # Split Dataset (Just TRAIN DATASET)
            X_train, X_test, y_train, y_test = self.featuresLabels(dfX1, dfy1)

            # train extractor (TRAIN DATA ONLY)
            self.fitFuncHV(X_train)
            # self.fitFuncTfidf(X_train)
            # self.fitFuncCV(X_train)
            # self.fitFuncW2V(X_train)
            # transform data
            X_train = self.transformFunc(X_train)
            X_test = self.transformFunc(X_test)

            # Machine Learning Time
            rf = self.randomForest(X_train, X_test, y_train, y_test)
            # ml = self.multilayerPerceptron(X_train, X_test, y_train, y_test)
            # sg = self.sgd(X_train, X_test, y_train, y_test)
            # ls = self.linearSVC(X_train, X_test, y_train, y_test)
            # sv = self.svc(X_train, X_test, y_train, y_test)
            # dt = self.decisionTree(X_train, X_test, y_train, y_test)
            # pe = self.perceptron(X_train, X_test, y_train, y_test)
            # kn = self.knn(X_train, X_test, y_train, y_test)
            # mn = self.multinomialNB(X_train, X_test, y_train, y_test)
            # gn = self.gaussianNB(X_train.toarray(), X_test.toarray(), y_train, y_test)

            results_rf.append(list(rf))
            # results_ml.append(list(ml))
            # results_sg.append(list(sg))
            # results_ls.append(list(ls))
            # results_sv.append(list(sv))
            # results_dt.append(list(dt))
            # results_pe.append(list(pe))
            # results_kn.append(list(kn))
            # results_mn.append(list(mn))
            # results_gn.append(list(gn))

        results_rf = np.mean(results_rf, axis=0)
        results_ml = np.mean(results_ml, axis=0)
        results_sg = np.mean(results_sg, axis=0)
        results_ls = np.mean(results_ls, axis=0)
        results_sv = np.mean(results_sv, axis=0)
        results_dt = np.mean(results_dt, axis=0)
        results_pe = np.mean(results_pe, axis=0)
        results_kn = np.mean(results_kn, axis=0)
        results_mn = np.mean(results_mn, axis=0)
        results_gn = np.mean(results_gn, axis=0)

        print("Classifier\tAccuracy\tPrecision\tRecall\tF1Score\tBalancedAccuracy")
        print("RandomForest\t{}\t{}\t{}\t{}\t{}".format(
            results_rf[0], results_rf[1], results_rf[2], results_rf[3], results_rf[4]))
        print("MLP\t{}\t{}\t{}\t{}\t{}".format(
            results_ml[0], results_ml[1], results_ml[2], results_ml[3], results_ml[4]))
        print("SGD\t{}\t{}\t{}\t{}\t{}".format(
            results_sg[0], results_sg[1], results_sg[2], results_sg[3], results_sg[4]))
        print("LinearSVC\t{}\t{}\t{}\t{}\t{}".format(
            results_ls[0], results_ls[1], results_ls[2], results_ls[3], results_ls[4]))
        print("SVCRBF\t{}\t{}\t{}\t{}\t{}".format(
            results_sv[0], results_sv[1], results_sv[2], results_sv[3], results_sv[4]))
        print("DecisionTree\t{}\t{}\t{}\t{}\t{}".format(
            results_dt[0], results_dt[1], results_dt[2], results_dt[3], results_dt[4]))
        print("Perceptron\t{}\t{}\t{}\t{}\t{}".format(
            results_pe[0], results_pe[1], results_pe[2], results_pe[3], results_pe[4]))
        print("KNN\t{}\t{}\t{}\t{}\t{}".format(
            results_kn[0], results_kn[1], results_kn[2], results_kn[3], results_kn[4]))
        print("MultinomialNB\t{}\t{}\t{}\t{}\t{}".format(
            results_mn[0], results_mn[1], results_mn[2], results_mn[3], results_mn[4]))
        print("GaussianNB\t{}\t{}\t{}\t{}\t{}".format(
            results_gn[0], results_gn[1], results_gn[2], results_gn[3], results_gn[4]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='.')

    parser.add_argument('--version', '-v', '-vvv', '-version',
                        action='version', version=str('Base 2.1'))

    parser.add_argument('--dataset-train', type=str, required=True,
                        help='This option define the dataset Train.')

    parser.add_argument('--dataset-test', type=str, required=True,
                        help='This option define the dataset Test.')

    # get args
    args = parser.parse_args()
    kwargs = {
        'dataset_train': args.dataset_train,
        'dataset_test': args.dataset_test
    }

    args = parser.parse_args()

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
