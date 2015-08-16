from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NNSentenceStructure
from Modules.ConsoleOutput import ConsoleOutput
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


    def loadTextFromFile(self, InputFile):
        ConsoleOutput.printGreen("Loading text data from: (" + InputFile + ")")
        sentence = []
        # Convert to natural language object
        for line in open(InputFile):
            # seperate punctuation from eachother so they have seprate tokens
            line = re.sub( r'(.)([,.!?:;"()\'\"])', r'\1 \2', line)
            # seperate from both directions
            line = re.sub( r'([,.!?:;"()\'\"])(.)', r'\1 \2', line)
            line = line.replace('"', '')
            sentence.extend(line.split())
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
            for identifier in NaturalLanguageObject._Identifiers:
                x = 10
        else:
            raise ValueError('Need to load data via loadFromTextFile() before calling function.')




    def __init__(self, inTrainRangeSS, inTrainRangeV):
        self._TrainingSequenceSS = []
        self._TrainingTargetsSS = []
        self._TrainRangeSS = inTrainRangeSS
        self._TrainRangeV = inTrainRangeV
