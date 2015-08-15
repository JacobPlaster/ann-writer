from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NeuralNetwork
import re

class NetworkTrainer:
    _TrainingSequence = []
    _TrainingTargets = []
    _TrainRange = 3

    def FeedFromFile(self, InputFile):
        print("Beginning training process...")
        sentence = []
        # Build training list
        for line in open(InputFile):
            # seperate punctuation from eachother so they have seprate tokens
            line = re.sub( r'(.)([,.!?:;"()\'\"])', r'\1 \2', line)
            # seperate from both directions
            line = re.sub( r'([,.!?:;"()\'\"])(.)', r'\1 \2', line)
            line = line.replace('"', '')
            sentence.extend(line.split())

        SentenceSize = len(sentence)
        # Break file into learnign sequences with defined targets
        for index in range(0, SentenceSize):
            trainSequence = []
            target = None
            if(index == len(sentence) - (self._TrainRange)):
                break
            for i in range(0, self._TrainRange+1):
                if(i == self._TrainRange):
                    target = sentence[index + i]
                    break
                trainSequence.append(sentence[index + i])
            tmpSequence = NaturalLanguageObject(trainSequence)
            tmpTarget = NaturalLanguageObject([target])

            self._TrainingSequence.append(tmpSequence.sentenceNormalised)
            self._TrainingTargets.extend(tmpTarget.sentenceNormalised)

        print("Data normalised successful...")
        return True

    def __init__(self, inTrainRange):
        self._TrainingSequence = []
        self._TrainingTargets = []
        self._TrainRange = inTrainRange
