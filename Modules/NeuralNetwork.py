import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from Modules.ConsoleOutput import ConsoleOutput

_MAX_DECIMAL_PLACES = 10

# This neural network is dedicated to figuring out the structure of the sentece
# And what comes next
class NNSentenceStructure:
    trainingData = []
    trainingDataResults = []
    clf = None


    # adds data to the training buffer
    def loadVectorsIntoNetwork(self, inNormalisedData, targetResult):
        self.trainingData.extend(inNormalisedData)
        self.trainingDataResults.extend(targetResult)

    # Fits the network to all of the data passed in
    def FitNetwork(self):
        countItems = len(self.trainingDataResults)

        self._fit(self.trainingData, self.trainingDataResults)

        ConsoleOutput.printGreen("Data successfully fitted to the sentence structure network.")
        ConsoleOutput.printGreen("Vectors: " + str(countItems))

        self.trainingData = None
        self.trainingDataResults = None


    def _fit(self, dataVector, targetVector):
        '''
        print('\n pf:')
        print(dataVector)
        print(targetVector)
        print('\n')
        '''
        self.clf.fit(np.asarray(dataVector, dtype="float"), np.asarray(targetVector, dtype="float"))

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        pred = self.clf.predict(inNormalisedData)
        return float(round(pred[0], _MAX_DECIMAL_PLACES))

    def getPredictionProbability(self, inNormalisedData):
        predProb = self.clf.predict_proba(inNormalisedData)
        return predProb


    def __init__(self):
        #self.clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
        # kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
        #self.clf = linear_model.SGDClassifier()

        # 26% accuracy
        self.clf = KNeighborsClassifier()

        # Bad accuracy (1%) but makes sense
        #self.clf = svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=0.0,
        #kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)



class NNVocabulary:
    trainingData = []
    trainingDataResults = []
    clf = None

    # adds data to the training buffer
    def loadVectorsIntoNetwork(self, inNormalisedData, targetResult):
        self.trainingData.extend(inNormalisedData)
        self.trainingDataResults.extend(targetResult)
        

    # Fits the network to all of the data passed in
    def FitNetwork(self):
        countItems = len(self.trainingDataResults)

        self._fit(self.trainingData, self.trainingDataResults)

        ConsoleOutput.printGreen("Data successfully fitted to the vocabulary network.")
        ConsoleOutput.printGreen("Vectors: " + str(countItems))

        self.trainingData = None
        self.trainingDataResults = None

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        pred = self.clf.predict(inNormalisedData)
        return float(round(pred[0], _MAX_DECIMAL_PLACES))

    def __init__(self):
        self.clf = KNeighborsClassifier()
