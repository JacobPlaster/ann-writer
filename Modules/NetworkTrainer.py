from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NNSentenceStructure
from Modules.ConsoleOutput import ConsoleOutput
from random import randint
import re

class NetworkTrainer:
    # Global training arrays
    # filled in the loadSentenceStructureNormals class
    _TrainingSequenceSS = []
    _TrainingTargetsSS = []
    # filled in the loadVocabularyNormals
    _TrainingSequenceV = []
    _TrainingTargetsV = []
    # Typical values
    _TrainRangeSS = 3
    _TrainRangeV = 2
    _nloTextData = None
    _Vocabulary = None


    def loadTextFromFile(self, InputFile):
        ConsoleOutput.printGreen("Loading text data from: (" + InputFile + ")")
        sentence = []
        # Convert to natural language object
        for line in open(InputFile):
            # remove completely
            line = line.replace('"', '')
            line = line.replace("'", '')
            # seperate punctuation from eachother so they have seprate tokens
            line = re.sub( r'(.)([,.!?:;"()\'\"])', r'\1 \2', line)
            # seperate from both directions
            line = re.sub( r'([,.!?:;"()\'\"])(.)', r'\1 \2', line)
            sentence.extend(line.split())
        ConsoleOutput.printGreen("Data load successful. WordCount: " + str(len(sentence)))
        self._nloTextData = NaturalLanguageObject(sentence)

    def loadSentenceStructureNormals(self):
        if(self._nloTextData != None):
            ConsoleOutput.printGreen("Beginning sentence structure parse...")

            SentenceSize = self._nloTextData.sentenceSize
            # Break file into learnign sequences with defined targets
            for index in range(0, SentenceSize):
                trainSequence = []
                target = None
                if(index == SentenceSize - (self._TrainRangeSS)):
                    break
                for i in range(0, self._TrainRangeSS+1):
                    # At the end of the sequence, so must be the target
                    if(i == self._TrainRangeSS):
                        target = self._nloTextData.sentenceNormalised[index + i]
                        break
                    trainSequence.append(self._nloTextData.sentenceNormalised[index + i])
                # Make sure we dont input the correct vector sizes into the neural network
                if(len(trainSequence) != self._TrainRangeSS):
                    raise ValueError('Train sequence vector not equal to _TrainRangeSS: ' + str(trainSequence))
                self._TrainingSequenceSS.append(trainSequence)
                self._TrainingTargetsSS.append(target)
            else:
                raise ValueError('Need to load data via loadFromTextFile() before calling function.')

        print("Data normalised successful...")
        return True

    def loadVocabularyNormals(self):
        if(self._nloTextData != None):
            ConsoleOutput.printGreen("Beginning sentence vocabulary parse...")
            # create vocabulary with the same amount of rows as the identifiers
            vocabulary = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
            # Build a vocabulary from the input data
            for wordIndex, x in enumerate(self._nloTextData.sentenceTokenList):
                wordToken = self._nloTextData.sentenceTokenList[wordIndex][1]
                word = self._nloTextData.sentenceTokenList[wordIndex][0]
                # find which colum to insert into
                for iIndex, iden in enumerate(NaturalLanguageObject._Identifiers):
                    # find colum
                    if(iden == wordToken):
                        #find if word already exists in rows
                        if word not in vocabulary[iIndex]:
                            vocabulary[iIndex].append(word)

            self._Vocabulary = vocabulary

            # print most populer
            #for index, i in enumerate(vocabulary):
                #print(NaturalLanguageObject._Identifiers[index] + " With " + str(len(vocabulary[index])))
        else:
            raise ValueError('Need to load data via loadFromTextFile() before calling function.')

    def getRandomWordFromIdentifier(self, indentifier):
        for index, val in enumerate(NaturalLanguageObject._Identifiers):
            if(indentifier == NaturalLanguageObject._Identifiers[index]):
                # return random word from dictionary
                return self._Vocabulary[index][randint(0, len(self._Vocabulary[index])-1)]




    def __init__(self, inTrainRangeSS, inTrainRangeV):
        self._TrainingSequenceSS = []
        self._TrainingTargetsSS = []
        self._TrainRangeSS = inTrainRangeSS
        self._TrainRangeV = inTrainRangeV
