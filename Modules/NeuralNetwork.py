import numpy as np
from sklearn import svm

class NeuralNetwork:
    isTraining = True
    trainingData = []
    trainingDataResults = []
    clf = None


    # adds data to the training buffer
    def trainNetwork(self, inNormalisedData, inTargetResult):
        self.trainingData.append(inNormalisedData)
        self.trainingDataResults.append(inTargetResult)

    # Fits the network to all of the data passed in
    def fitNetwork(self):
        self.clf.fit(self.trainingData, self.trainingDataResults)
        print(self.trainingData)
        # free up unneccassary resources
        self.trainingData = None
        self.trainingDataResults = None

    # gets a prediction from the network with the given input
    def getPrediction(self, inNormalisedData):
        return self.clf.predict(inNormalisedData)

    def __init__(self):
        self.isTraining = True
        self.clf = svm.SVC(kernel='linear', C=1.0)
