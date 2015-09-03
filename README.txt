Coming soon...

if(val == "-utss"):
        _isUnitTestingSS = True
    # Unit testing for the vocabulary network
    elif(val == "-utv"):
        _isUnitTestingV = True
    # Allows for the recursive user input loop to run
    elif(val == "-ri"):
        _recursiveInput = True
    elif(len(consoleInArgs) >= index+1):
        if(val == "-td"):
            _TrainingDataInputFile = consoleInArgs[index+1]
            ConsoleOutput.printGreen("Training data load locaiton changed to: \"" + _TrainingDataInputFile + "\"")
        elif(val == "-ts"):

### Artificial Neural-network writer
A neural network that utilises sk-learn, numpy and nltk in an attempt to generate text in the style of any given training data.
<br>
<br>
### Installing
install instructions here...
<br>
<br>
### Usage
Run the program:
''' bash
python3 main.py
'''
Run the unit test function on the sentence structuring:
''' bash
python3 main.py -utss
'''
Unit test function for vocabulary:
''' bash
python3 main.py -utv
'''
Specify the training data file:
''' bash
python3 main.py -td <filepath>
'''
Specify test sentence: (Generates text that follows on from the input)
example input = "the boy ran"
''' bash
python3 main.py -ts "<input sentence here>"
'''
Specify the number of words generated for given test sentence:
''' bash
python3 main.py -tsc <genCount>
'''
<br>
Example usage scenario:
''' bash
python3 main.py -ts "today i will" -tsc 10 -td "Datasets/HarryPotter(xxlarge).txt"
'''

to read more go to my (blog post.)[http://www.jacobplaster.net/artificial-neural-network-writer]
