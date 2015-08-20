'''
    write artificial neural network will read attempt to impersonate the personality and
    writing style of the publisher who wrote the text that the netwrok was trained on.

    We will try both shakespear, J.K Rowling and random stuff from the internet

    The neural network will:
    - Increment sentence by sentence through the training data
    - Convert sentence into parsed token sentence using nltk
    - train the sequences of the tokens

    - Increment sentence by sentece again
    - Train the probability of each token appearing (probability of verb being 'is')

    Most programs that attempt to predict text usually measure the probability of characters occuring, however
    this is very inaccurate. This method will always produce good results.
'''
import sys, getopt
from _thread import *
from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NNSentenceStructure, NNVocabulary
from Modules.NetworkTrainer import NetworkTrainer
from Modules.UnitTesting import UnitTester
from colorama import init, deinit

_TrainingDataInputFile = "Datasets/MacbookAirBlog(x3576).txt"
#_TrainingDataInputFile = "Datasets/HarryPotter(x1737).txt"
# Amount of vectors per a train statement
_TrainRangeSS = 3
_TrainRangeV = 1

def Main():
    _isUnitTesting = False
    _recursiveInput = False
    consoleInArgs = sys.argv[1:]
    # check input arguments
    for index, val in enumerate(consoleInArgs):
        # Runs the unit testing module on initiation
        if(val == "-ut"):
            _isUnitTesting = True
        # Allows for the recursive user input loop to run
        elif(val == "-ri"):
            _recursiveInput = True
        else:
            raise ValueError('Un-recognized console argument: ' + val)
    # Initialise colorama cross-platform console logging
    init()

    neuralNetworkSS = NNSentenceStructure()
    neuralNetworkV = NNVocabulary()
    # Network trainer converts text data into normalized vectors that
    # can be passed into the networks
    networkTrainer = NetworkTrainer(_TrainRangeSS, _TrainRangeV)
    networkTrainer.loadTextFromFile(_TrainingDataInputFile)
    # Trainer parses the structure into vector normal arrays of size (_TrainRangeSS)
    # the next word of the squence is used as the target, example
    # ["Harry", "sat", "on", "his"] - ["broomstick"] <-- target
    networkTrainer.loadSentenceStructureNormals()
    networkTrainer.loadVocabularyNormals()
    # Pass the vectors into the network
    neuralNetworkSS.loadVectorsIntoNetwork(networkTrainer._TrainingSequenceSS, networkTrainer._TrainingTargetsSS)
    # Passs into vocab network here ****

    # Fit data
    neuralNetworkSS.FitNetwork()
    # Fit to vocab network here ****

    # Use console argument "-ut" to activate
    if(_isUnitTesting):
        #testing
        uTester = UnitTester(neuralNetworkSS, _TrainRangeSS)
        uTester.TestSentenceStructuring()
        '''
        tmpNl = NaturalLanguageObject(['looked', 'at', 'ron', 'and'])
        print('\n')
        print('Input: ' + str(tmpNl.sentenceTokenList))
        print('Prediction: ' + str(tmpNl.tokeniseNormals(neuralNetwork.getPrediction(tmpNl.sentenceNormalised))))
        print('\n')
        '''

    # Use console argument "-ri" to activate
    if(_recursiveInput):
        while(True):
            inputIn = input("Enter " + str(_TrainRangeSS) + " words: ")
            inputSen = inputIn.split()
            if(len(inputSen) == _TrainRangeSS):
                nlO = NaturalLanguageObject(inputSen)
                print(str(nlO.sentenceTokenList))
                testPred = neuralNetworkSS.getPrediction(nlO.sentenceNormalised)
                testPred = nlO.tokeniseNormals([testPred])
                print("Predicted: " + str(testPred))
                print(inputIn + " " + str(networkTrainer.getRandomWordFromIdentifier(testPred[0])))
            else:
                print("Testing requires an input range of: " + str(_TrainRangeSS))

    genSize = 30
    initialInput = "The Macbook air"
    print(initialInput + " ", end="")
    initialInput = initialInput.split()
    # generate a sentence of genSize
    for index in range(0, genSize):
        nlO = NaturalLanguageObject(initialInput)
        testPred = neuralNetworkSS.getPrediction(nlO.sentenceNormalised)
        testPred = nlO.tokeniseNormals([testPred])
        word = str(networkTrainer.getRandomWordFromIdentifier(testPred[0]))
        print("" + word + " ", end="")
        del initialInput[0]
        initialInput.append(word)
    print("\n")
    # Reset console back to original state
    deinit()

if __name__ == '__main__':
    Main()
