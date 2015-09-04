from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.MachineLearning import NNSentenceStructure
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
    _TrainRangeV = 1
    _nloTextData = None


    def loadTextFromFile(self, InputFile):
        ConsoleOutput.printGreen("Loading text data from: (" + InputFile + ")")
        sentence = []
        # Convert to natural language object
        for line in open(InputFile):
            #line = line.lower()
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
                # Make sure we dont input the correct vector sizes into the network
                if(len(trainSequence) != self._TrainRangeSS):
                    raise ValueError('Train sequence vector not equal to _TrainRangeSS: ' + str(trainSequence))
                self._TrainingSequenceSS.append(trainSequence)
                self._TrainingTargetsSS.append(target)
            else:
                raise ValueError('Need to load data via loadFromTextFile() before calling function.')

        ConsoleOutput.printGreen("Data normalised successful...")
        return True

    def loadVocabularyNormals(self, NNV):
        if(self._nloTextData != None):
            ConsoleOutput.printGreen("Beginning sentence vocabulary parse...")
            # create vocabulary with the same amount of rows as the identifiers
            vocabulary = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
            tempNonUniqueVocab = [list() for _ in range(len(NaturalLanguageObject._Identifiers))]
            # Build a vocabulary from the input data
            # all elements apart from first few
            for wordIndex, x in enumerate(self._nloTextData.sentenceTokenList[self._TrainRangeV:]):
                wordToken = self._nloTextData.sentenceTokenList[wordIndex][1]
                word = self._nloTextData.sentenceTokenList[wordIndex][0]
                prevTokenNormal = self._nloTextData.sentenceNormalised[wordIndex-1]
                # find which colum to insert into
                for iIndex, iden in enumerate(NaturalLanguageObject._Identifiers):
                    # find colum
                    if(iden == wordToken):
                        #find if combination of identifier and word already exist
                        if (prevTokenNormal, word) not in vocabulary[iIndex]:
                            # unique sequences will be stored in the vocabulary for lookups
                            # when converting from normals back into words
                            vocabulary[iIndex].append((prevTokenNormal, word))
                        else:
                            # get the non-unique combinations (purely for training)
                            tempNonUniqueVocab[iIndex].append((prevTokenNormal, word))
            # Use unique sequences to generate normals
            for index, val in enumerate(vocabulary):
                # Calculate the normals for each row
                normalisedUnit = 0
                if(len(vocabulary[index])>0):
                    normalisedUnit = 2/len(vocabulary[index])
                for index2, vector in enumerate(vocabulary[index]):
                    tmpNormal = round(float(((index2+1) * normalisedUnit)), 10)
                    word = vector[1]
                    prevNormal = vector[0]
                    # pass into the network fit buffer (THESE ARE THE UNIQUE COMBINATIONS)
                    NNV.loadVectorsIntoNetworkByIndex(index, prevNormal, tmpNormal)
                    NNV.loadVocab(index, tmpNormal, word)
                    # check non-unique for same sequence
                    for iNU, nonUniqueVal in enumerate(tempNonUniqueVocab[index]):
                        # if there are non-unique sequences then add to training
                        if (prevNormal, word) == tempNonUniqueVocab[index][iNU]:
                            NNV.loadVectorsIntoNetworkByIndex(index, prevNormal, tmpNormal)
                            NNV.loadVocab(index, tmpNormal, word)
        else:
            raise ValueError('Need to load data via loadFromTextFile() before calling function.')


    def __init__(self, inTrainRangeSS, inTrainRangeV):
        self._TrainingSequenceSS = []
        self._TrainingTargetsSS = []
        self._TrainRangeSS = inTrainRangeSS
        self._TrainRangeV = inTrainRangeV
