from Modules.NaturalLanguage import NaturalLanguageObject
from Modules.NeuralNetwork import NeuralNetwork

class NetworkTrainer:

    def FeedToInputFileToNetwork(NeuralNetwork, InputFile):
        _TrainRange = 3

        print("Beginning training process...")
        sentence = []
        # Build training list
        for line in open(InputFile):
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
            print(tmpSequence.sentence)
            print(target)
        return True
