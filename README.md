# Artificial Neural-network writer
A neural network that utilises sk-learn, numpy and nltk in an attempt to generate text in the style of any given training data. Written in python3.
<br>
<br>
### Installing
```bash
pip3 install -U scikit-learn
pip3 install -U numpy
pip3 install -U nltk
```
<br>
<br>
### Usage
Run the program:
```bash
python3 main.py
```
Run the unit test function on the sentence structuring:
```bash
python3 main.py -utss
```
Unit test function for vocabulary:
```bash
python3 main.py -utv
```
Specify the training data file:
```bash
python3 main.py -td <filepath>
```
Specify test sentence: (Generates text that follows on from the input)
example input = "the boy ran"
```bash
python3 main.py -ts "<input sentence here>"
```
Specify the number of words generated for given test sentence:
```bash
python3 main.py -tsc <genCount>
```
<br>
Example usage scenario:
```bash
python3 main.py -ts "today i will" -tsc 10 -td "Datasets/HarryPotter(xxlarge).txt"
```
<br>
<br>

To read more go to my [blog post](http://www.jacobplaster.net/artificial-neural-network-writer). Here I show multiple text generations with different training data sets and how accurate the program is at impersonating the training data.
