from Modules.NeuralNetwork import NeuralNetwork
from Modules.NaturalLanguage import NaturalLanguageObject

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
                prediction = str(nloTester.tokeniseNormals(self.neuralNetwork.getPrediction(nloTester.sentenceNormalised)))
                comp = str(nloTarget.sentenceTags)

                if(prediction == comp):
                    passedTests.append("Phrase: " + str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: " + prediction)
                else:
                    failedTests.append("Phrase: " + str(nloTester.sentenceTokenList) + "   Target: " + str(nloTarget.sentenceTokenList) + "    Prediction: " + prediction)

        # print results
        print("\n")
        print("********** TestSentenceStructuring() **********")
        print("\nFailed Tests: (" + str(len(failedTests)) + "/" + str(len(testingPara)) + ")")
        for val in failedTests:
            print(val)
        print("\nPassed Tests: (" + str(len(passedTests)) + "/" + str(len(testingPara)) + ")")
        for val in passedTests:
            print(val)


    def __init__(self, inNeuralNetwork, inVectorSize):
        self.neuralNetwork = inNeuralNetwork
        self.VectorSize = inVectorSize
