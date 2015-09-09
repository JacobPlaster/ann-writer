'''
    write artificial machine learning will read attempt to impersonate the personality and
    writing style of the publisher who wrote the text that the netwrok was trained on.

    We will try both shakespear, J.K Rowling and random stuff from the internet

    The program will:
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
from Modules.MachineLearning import NNSentenceStructure, NNVocabulary
from Modules.NetworkTrainer import NetworkTrainer
from Modules.UnitTesting import UnitTester
from colorama import init, deinit
from Modules.ConsoleOutput import ConsoleOutput

# Amount of vectors per a train statement
_TrainRangeSS = 3
_TrainRangeV = 1

def Main():
    _isUnitTestingSS = False
    _isUnitTestingV = False
    _recursiveInput = False
    _TrainingDataInputFile = "Datasets/HarryPotter(xxlarge).txt"
    _TestSentence = ""
    _TestSequenceGenSize = 30
    _OutputFile = None

    consoleInArgs = sys.argv[1:]
    # check input arguments
    for index, val in enumerate(consoleInArgs):
        # Runs the unit testing module on initiation
        if(val == "-utss"):
            _isUnitTestingSS = True
        # Unit testing for the vocabulary network
        elif(val == "-utv"):
            _isUnitTestingV = True
        elif(len(consoleInArgs) >= index+1):
            # specify training data location
            if(val == "-td"):
                _TrainingDataInputFile = consoleInArgs[index+1]
                ConsoleOutput.printGreen("Training data load locaiton changed to: \"" + _TrainingDataInputFile + "\"")
            # give a generation sentence input
            elif(val == "-ts"):
                _TestSentence = consoleInArgs[index+1]
                if(len(_TestSentence.split()) != _TrainRangeSS):
                    raise ValueError('Test sequence must be the same length as the vector training size. (' + str(_TrainRangeSS) + ')')
            # set the amount of words generated after input
            elif(val == "-tsc"):
                _TestSequenceGenSize = int(consoleInArgs[index+1])
                ConsoleOutput.printGreen("Test sequence generation size changed to: " + str(_TestSequenceGenSize))
            # set the output file for the generated data to be printed to
            elif(val == "-of"):
                _OutputFile = str(consoleInArgs[index+1])
                ConsoleOutput.printGreen("Output generation location changed to: (" + consoleInArgs[index+1]+ ")")
        else:
            raise ValueError('Un-recognized console argument: ' + str(val))
    # Initialise colorama cross-platform console logging
    init()

    MLNetworkSS = NNSentenceStructure()
    MLNetworkV = NNVocabulary()
    # Network trainer converts text data into normalized vectors that
    # can be passed into the networks
    networkTrainer = NetworkTrainer(_TrainRangeSS, _TrainRangeV)
    networkTrainer.loadTextFromFile(_TrainingDataInputFile)
    # Trainer parses the structure into vector normal arrays of size (_TrainRangeSS)
    # the next word of the squence is used as the target, example
    # ["Harry", "sat", "on", "his"] - ["broomstick"] <-- target
    networkTrainer.loadSentenceStructureNormals()
    networkTrainer.loadVocabularyNormals(MLNetworkV)
    # Pass the vectors into the network
    MLNetworkSS.loadVectorsIntoNetwork(networkTrainer._TrainingSequenceSS, networkTrainer._TrainingTargetsSS)
    # Passs into vocab network here ****

    # Fit data
    MLNetworkSS.FitNetwork()
    MLNetworkV.FitNetwork()
    # Fit to vocab network here ****

    # Use console argument "-utss" to activate
    #testing
    uTester = None
    if(_isUnitTestingSS):
        if(uTester == None):
            uTester = UnitTester(MLNetworkSS, MLNetworkV, _TrainRangeSS, _TrainRangeV)
        uTester.TestSentenceStructuring()
    # use console argument "-utv" to activate
    if(_isUnitTestingV):
        if(uTester == None):
            uTester = UnitTester(MLNetworkSS, MLNetworkV, _TrainRangeSS, _TrainRangeV)
        uTester.TestVocabulary()

    if(_TestSentence != ""):
        printToFile = False
        f = None
        # user has specified output location
        if(_OutputFile != None):
            printToFile = True
            f = open(_OutputFile,'w')
        genSize = _TestSequenceGenSize
        initialInput = _TestSentence
        if(printToFile):
            f.write(initialInput + " ")
        else:
            print(initialInput + " ", end="")
        initialInput = initialInput.split()
        # generate a sentence of genSize
        for index in range(0, genSize):
            nlo = NaturalLanguageObject(initialInput)
            # since nlo will always be the right size, we can use that variable
            predToke = MLNetworkSS.getPrediction(nlo.sentenceNormalised)
            nextToke = nlo.tokeniseNormals([predToke])
            # now we have the next toke in the sentence, convert that to word
            word = MLNetworkV.getPredictedWord(nlo.sentenceNormalised[-1], nextToke[0])
            # decide whether to print to file or console
            if(printToFile):
                f.write(str(word) + " ")
            else:
                print(str(word) + " ", end="")
            initialInput.append(word)
            # maintain a size of 'genSize'
            del initialInput[0]
        print("\n")
    # Reset console back to original state
    deinit()

if __name__ == '__main__':
    Main()
