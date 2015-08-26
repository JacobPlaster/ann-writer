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
from Modules.ConsoleOutput import ConsoleOutput

_TrainingDataInputFile = "Datasets/MacbookAirBlog(x3576).txt"
#_TrainingDataInputFile = "Datasets/HarryPotter(x4546).txt"
# Amount of vectors per a train statement
_TrainRangeSS = 3
_TrainRangeV = 1

def Main():
    _isUnitTestingSS = False
    _isUnitTestingV = False
    _recursiveInput = False
    consoleInArgs = sys.argv[1:]
    # check input arguments
    for index, val in enumerate(consoleInArgs):
        # Runs the unit testing module on initiation
        if(val == "-utss"):
            _isUnitTestingSS = True
        # Unit testing for the vocabulary network
        elif(val == "-utv"):
            _isUnitTestingV = True
        # Allows for the recursive user input loop to run
        elif(val == "-ri"):
            _recursiveInput = True
        elif(len(consoleInArgs) >= index+1):
            if(val == "-td"):
                _TrainingDataInputFile = consoleInArgs[index+1]
                ConsoleOutput.printGreen("Training data load locaiton changed to: \"" + _TrainingDataInputFile + "\"")
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
    networkTrainer.loadVocabularyNormals(neuralNetworkV)
    # Pass the vectors into the network
    neuralNetworkSS.loadVectorsIntoNetwork(networkTrainer._TrainingSequenceSS, networkTrainer._TrainingTargetsSS)
    # Passs into vocab network here ****

    # Fit data
    neuralNetworkSS.FitNetwork()
    neuralNetworkV.FitNetwork()
    # Fit to vocab network here ****

    # Use console argument "-utss" to activate
    #testing
    uTester = None
    if(_isUnitTestingSS):
        if(uTester == None):
            uTester = UnitTester(neuralNetworkSS, neuralNetworkV, _TrainRangeSS, _TrainRangeV)
        uTester.TestSentenceStructuring()
    # use console argument "-utv" to activate
    if(_isUnitTestingV):
        if(uTester == None):
            uTester = UnitTester(neuralNetworkSS, neuralNetworkV, _TrainRangeSS, _TrainRangeV)
        uTester.TestVocabulary()

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
    initialInput = "why dont we"
    print(initialInput + " ", end="")
    initialInput = initialInput.split()
    # generate a sentence of genSize
    for index in range(0, genSize):
        nlo = NaturalLanguageObject(initialInput)
        # since nlo will always be the right size, we can use that variable
        predToke = neuralNetworkSS.getPrediction(nlo.sentenceNormalised)
        nextToke = nlo.tokeniseNormals([predToke])
        # now we have the next toke in the sentence, convert that to word
        word = neuralNetworkV.getPredictedWord(nlo.sentenceNormalised[-1], nextToke[0])
        print(str(word) + " ", end="")
        initialInput.append(word)
        # maintain a size of 'genSize'
        del initialInput[0]
    print("\n")
    # Reset console back to original state
    deinit()

if __name__ == '__main__':
    Main()
