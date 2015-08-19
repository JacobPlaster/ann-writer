from Modules.NeuralNetwork import NNSentenceStructure
from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.ConsoleOutput import ConsoleOutput
from sklearn.metrics import accuracy_score
import numpy as np

testingParaMacbookBlog = ['The', '12-inch', 'Retina', 'MacBook', 'is', 'Apple', "'", 's',
 'latest', 'and', 'greatest', 'notebook', ',', 'and', 'will', 'very', 'likely',
  'replace', 'the', 'MacBook', 'Air', 'entirely', 'once', 'Apple', 'is', 'able',
   'to', 'bring', 'its', 'costs', 'down', 'enough', ',', 'though', 'this', 'may',
    'take', 'a', 'few', 'generations', '.', 'It', "'", 's', 'fresh', 'on', 'the',
     'market', ',', 'having', 'been', 'released', 'on', 'April', '10', ',', 'and',
      'it', 'features', 'all', 'of', 'Apple', "'", 's', 'newest', 'technology', '.',
       'Like', 'to', 'have', 'the', 'coolest', 'product', 'on', 'the', 'market', '?',
        'This', 'is', 'it', '.', 'Looking', 'for', 'the', 'most', 'portable', 'Apple',
         'notebook', '?', 'You', 'found', 'it', '.', 'Want', 'to', 'wow', 'your', 'friends',
          '?', 'You', 'need', 'a', 'Retina', 'MacBook', '.']
testingParaHarryPotter = ['He', 'stopped', 'there', 'to', 'enjoy', 'the', 'effect', 'of',
 'these', 'words', '.', 'He', 'could', 'almost', 'see', 'the', 'cogs', 'working',
  'under', 'Uncle', 'Vernon’s', 'thick', ',', 'dark', ',', 'neatly', 'parted',
   'hair', '.', 'If', 'he', 'tried', 'to', 'stop', 'Harry', 'writing', 'to', 'Sirius',
    ',', 'Sirius', 'would', 'think', 'Harry', 'was', 'being', 'mistreated', '.',
     'If', 'he', 'told', 'Harry', 'he', 'couldn’t', 'go', 'to', 'the', 'Quidditch', 'World',
      'Cup', ',', 'Harry', 'would', 'write', 'and', 'tell', 'Sirius', ',', 'who', 'would',
       'know', 'Harry', 'was', 'being', 'mistreated', '.', 'There', 'was', 'only', 'one',
        'thing', 'for', 'Uncle', 'Vernon', 'to', 'do', '.', 'Harry', 'could', 'see', 'the',
         'conclusion', 'forming', 'in', 'his', 'uncle’s', 'mind', 'as', 'though', 'the',
          'great', 'mustached', 'face', 'were', 'transparent', '.', 'Harry', 'tried', 'not',
           'to', 'smile', ',', 'to', 'keep', 'his', 'own', 'face', 'as', 'blank', 'as', 'possible', '.']

class UnitTester:
    neuralNetwork = None
    VectorSize = 3

    def TestVocabulary():
        print("Testing Vocabulary...")

    def TestSentenceStructuring(self):

        #testingPara = testingParaHarryPotter
        testingPara = testingParaMacbookBlog
        passedTests = []
        nonFatalTests = []
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
                    #filter for probability
                    probList = self.neuralNetwork.getPredictionProbability(nloTester.sentenceNormalised)
                    prob = 0
                    for val in probList[0]:
                        if(val > prob):
                            prob = val
                    passedTests.append(str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: "
                    + prediction  + " " +str(prob*100) + "%")
                else:
                    probList = self.neuralNetwork.getPredictionProbability(nloTester.sentenceNormalised)
                    prob = 0
                    for val in probList[0]:
                        if(val > prob):
                            prob = val
                    # if accuracy s less than 30% add to failed list
                    if(prob < 0.3):
                        failedTests.append(str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: "
                        + prediction  + " " +str(prob*100) + "%")
                    else:
                        # if probability is more than 60% its probably passed
                        if(prob > 0.6):
                            passedTests.append(str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: "
                            + prediction  + " " +str(prob*100) + "%")
                        else:
                            nonFatalTests.append(str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: "
                            + prediction  + " " +str(prob*100) + "%")

        # print results
        print("\n")
        print("********** TestSentenceStructuring() **********")
        print("\n")
        ConsoleOutput.printUnderline("Failed Tests: (" + str(len(failedTests)) + "/" + str(len(testingPara)) + ")")
        for val in failedTests:
            ConsoleOutput.printRed(val)
        print("\n")
        ConsoleOutput.printUnderline("Non-Fatal failed Tests: (" + str(len(nonFatalTests)) + "/" + str(len(testingPara)) + ")")
        for val in nonFatalTests:
            ConsoleOutput.printYellow(val)
        print("\n")
        ConsoleOutput.printUnderline("Passed Tests: (" + str(len(passedTests)) + "/" + str(len(testingPara)) + ")")
        for val in passedTests:
            ConsoleOutput.printGreen(val)
        print("\n")

        nnAccuracy = accuracy_score(np.array(acTestTrue).astype(int), np.array(acTestPred).astype(int))
        ConsoleOutput.printYellow("Passed: " + str(len(passedTests)) + "   Non-Fatals: " + str(len(nonFatalTests)) + "   Fails: " + str(len(failedTests)))
        ConsoleOutput.printYellow("NeuralNetork accuracy: " + str(round(nnAccuracy*100,1)) + "%")
        print("\n")


    def __init__(self, inNeuralNetwork, inVectorSize):
        self.neuralNetwork = inNeuralNetwork
        self.VectorSize = inVectorSize
