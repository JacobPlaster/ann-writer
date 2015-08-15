import numpy as np
from sklearn import svm
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

class NeuralNetwork:
    isTraining = True
    trainingData = []
    trainingDataResults = []
    clf = None
    isFirstTimeFitting = True
    _BUFFER_AMOUNT = 100


    # adds data to the training buffer
    def loadVectorsIntoNetwork(self, inNormalisedData, targetResult):
        self.trainingData.extend(inNormalisedData)
        self.trainingDataResults.extend(targetResult)

    # Fits the network to all of the data passed in
    def FitNetwork(self):
        countItems = len(self.trainingDataResults)

        self._fit(self.trainingData, self.trainingDataResults)

        print("Data successfully fitted to the network.")
        print("Vectors: " + str(countItems))

        self.trainingData = None
        self.trainingDataResults = None


    def _fit(self, dataVector, targetVector):
        '''
        print('\n pf:')
        print(dataVector)
        print(targetVector)
        print('\n')
        '''
        self.clf.fit(np.asarray(dataVector, dtype="complex"), np.asarray(targetVector, dtype="complex"))

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        return self.clf.predict(inNormalisedData)

    def __init__(self):
        self.isTraining = True
        #self.clf = svm.SVC(kernel='linear', C=1.0)
        #self.clf = linear_model.SGDClassifier()
        self.clf = KNeighborsClassifier()
