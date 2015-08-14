import numpy as np
from sklearn import svm
from sklearn import linear_model

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
        '''
        # Buffer loads data incrementally into the network
        if(len(self.trainingData) < self._BUFFER_AMOUNT):
            # No need to buffer, dataset not large enough
            self._fit(self.trainingData, self.trainingDataResults)
            self.trainingData = []
            self.trainingDataResults = []
            countIterations = countIterations+1
        else:
            # Dataset too large, begin to buffer in 10 at a time
            while(len(self.trainingData) >= self._BUFFER_AMOUNT):
                countIterations = countIterations+1
                # Coming to the end of the dataset
                if(len(self.trainingData) < self._BUFFER_AMOUNT):
                    self._fit(self.trainingData, self.trainingDataResults)
                    self.trainingData = []
                    self.trainingDataResults = []
                    break
                self._fit(self.trainingData[:self._BUFFER_AMOUNT], self.trainingDataResults[:self._BUFFER_AMOUNT])
                del self.trainingData[:self._BUFFER_AMOUNT]
                del self.trainingDataResults[:self._BUFFER_AMOUNT]

        # free up unneccassary resoefdgFDG
        self.trainingData = None
        self.trainingDataResults = None

        print("Data successfully fitted to the network.")
        print("Vectors: " + str(countItems) + "   Iterations: " + str(countIterations) + "   BufferSize: " + str(self._BUFFER_AMOUNT))
        '''

    def _fit(self, dataVector, targetVector):
        '''
        print('\n pf:')
        print(dataVector)
        print(targetVector)c
        print('\n')
        '''
        self.clf.fit(np.asarray(dataVector, dtype="complex"), np.asarray(targetVector, dtype="complex"))

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        return self.clf.predict(inNormalisedData)

    def __init__(self):
        self.isTraining = True
        #self.clf = svm.SVC(kernel='linear', C=1.0)
        self.clf = linear_model.SGDClassifier()
