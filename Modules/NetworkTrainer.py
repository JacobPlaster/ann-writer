from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NeuralNetwork
import re

class NetworkTrainer:
    _TrainingSequence = []
    _TrainingTargets = []

    def FeedFromFile(self, InputFile):
        _TrainRange = 3

        print("Beginning training process...")
        sentence = []
        # Build training list
        for line in open(InputFile):
            # seperate punctuation from eachother so they have seprate tokens
            line = re.sub( r'([a-zA-Z])([,.!?:;"()\'\"])', r'\1 \2', line)
            line = line.replace('"', '')
            sentence.extend(line.split())

        SentenceSize = len(sentence)
        # Break file into learnign sequences with defined targets
        for index in range(0, SentenceSize):
            trainSequence = []
            target = None
            if(index == len(sentence) - (_TrainRange)):
                break
            for i in range(0, _TrainRange+1):
                if(i == _TrainRange):
                    target = sentence[index + i]
                    break
                trainSequence.append(sentence[index + i])
            tmpSequence = NaturalLanguageObject(trainSequence)
            tmpTarget = NaturalLanguageObject([target])

            self._TrainingSequence.append(tmpSequence.sentenceNormalised)
            self._TrainingTargets.extend(tmpTarget.sentenceNormalised)

        print("Data normalised successful...")
        return True

    def __init__(self):
        self._TrainingSequence = []
        self._TrainingTargets = []
