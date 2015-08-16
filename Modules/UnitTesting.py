from Modules.NeuralNetwork import NNSentenceStructure
from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.ConsoleOutput import ConsoleOutput
from sklearn.metrics import accuracy_score
import numpy as np

class UnitTester:
    neuralNetwork = None
    VectorSize = 3

    def TestVocabulary():
        print("Testing Vocabulary...")

    def TestSentenceStructuring(self):
        testingPara = ['A', 'boy', 'in', 'pale', 'blue', 'robes', 'jumped', 'down', 'from', 'the',
                        'carriage', ',', 'bent', 'forward', ',', 'fumbled', 'for', 'a', 'moment',
                         'with', 'something', 'on', 'the', 'carriage', 'floor', ',', 'and', 'unfolded',
                          'a', 'set', 'of', 'golden', 'steps', '.', 'He', 'sprang', 'back', 'respectfully',
                           '.', 'Then', 'Harry', 'saw', 'a', 'shining', ',', 'high-heeled', 'black', 'shoe',
                            'emerging', 'from', 'inside', 'of', 'the', 'carriage', '–', 'a', 'shoe', 'the',
                             'size', 'of', 'a', 'child’s', 'sled', '–', 'followed', ',', 'almost', 'immediately',
                              ',', 'by', 'the', 'largest', 'woman', 'he', 'had', 'ever', 'seen', 'in', 'his',
                              'life', '.', 'The', 'size', 'of', 'the', 'carriage', ',', 'and', 'of', 'the', 'horses',
                               ',', 'was', 'immediately', 'explained', '.', 'A', 'few', 'people', 'gasped', '.']
        passedTests = []
        failedTests = []
        # used to predict accuracy of the network
        acTestPred = []
        acTestTrue = []

        # Build a test sequence form each word
        for index, val in enumerate(testingPara):
            tmpTestSeq = []
            target = None
            # grab the next 3 words after
            if(index < len(testingPara)-(self.VectorSize+1)):
                for index2 in range(0, self.VectorSize):
                    tmpTestSeq.append(testingPara[index+index2])
                target = testingPara[index+self.VectorSize]
                # convert to natural language object
                nloTester = NaturalLanguageObject(tmpTestSeq)
                nloTarget = NaturalLanguageObject([target])
                # get nerual network prediction
                normalPred = self.neuralNetwork.getPrediction(nloTester.sentenceNormalised)
                prediction = str(nloTester.tokeniseNormals([normalPred]))
                comp = str(nloTarget.sentenceTags)

                cTrue = nloTarget.sentenceNormalised[0]
                acTestTrue.append(cTrue*100)
                acTestPred.append(normalPred*100)

                #if first letters match, this means 'NN' will match with 'NNS'
                if(prediction[2] == comp[2]):
                    passedTests.append("Phrase: " + str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: " + prediction)
                    #passedTests.append(self.neuralNetwork.accuracy_score(nloTester.sentenceNormalised, predicted))
                else:
                    failedTests.append("Phrase: " + str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: " + prediction)

        # print results
        print("\n")
        print("********** TestSentenceStructuring() **********")
        print("\n")
        ConsoleOutput.printUnderline("Non-Fatal failed Tests: (" + str(len(failedTests)) + "/" + str(len(testingPara)) + ")")
        for val in failedTests:
            ConsoleOutput.printYellow(val)
        print("\n")
        ConsoleOutput.printUnderline("Passed Tests: (" + str(len(passedTests)) + "/" + str(len(testingPara)) + ")")
        for val in passedTests:
            ConsoleOutput.printGreen(val)
        print("\n")

        nnAccuracy = accuracy_score(np.array(acTestTrue).astype(int), np.array(acTestPred).astype(int))
        ConsoleOutput.printYellow("Passed: " + str(len(passedTests)) + "   Non-Fatals:" + str(len(failedTests)) + "   Fails: 0")
        ConsoleOutput.printYellow("NeuralNetork accuracy: " + str(round(nnAccuracy*100,1)) + "%")
        print("\n")


    def __init__(self, inNeuralNetwork, inVectorSize):
        self.neuralNetwork = inNeuralNetwork
        self.VectorSize = inVectorSize
