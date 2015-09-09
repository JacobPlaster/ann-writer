# Machine learning text generator
Machine learning that utilises sk-learn, numpy and nltk in an attempt to generate text in the style of any given training data. Written in python3.
<br>
<br>
### Installing
------
```bash
pip3 install -U scikit-learn
pip3 install -U numpy
pip3 install -U nltk
```
<br>
### Usage
------
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
Output generated text to a file:
```bash
python3 main.py -of "<fileLocation>"
```
<br>
Example usage scenario:
```bash
python3 main.py -ts "today i will" -tsc 10 -td "Datasets/HarryPotter(xxlarge).txt"
```
<br>

### Datasets
------
Includes 6 datasets:
```bash
HarryPotter(small).txt = 346 training vectors
HarryPotter(medium).txt = 2500 training vectors
HarryPotter(large).txt = 4550 training vectors
HarryPotter(xlarge).txt = 11429 training vectors
HarryPotter(xxlarge).txt = 15829 training vectors

MacbookAirBlog(large).txt = 3576 training vectors
```
Change the data sets with the '-td' command. The larger the data set, the longer the program will take to fit and produce a result. The ability to load an already fitted network has not been implemented yet, so the program has to run the initial fit every time.<br>
The Harry potter data sets have been taken from the book directly and the macbook dataset was taken from a random blog.
<br>
It is extremely easy to add your own data set, just make sure that it is in the form of a text blob (see provided datasets). And then simply use the command line to select your dataset
```bash
python3 main.py -td "Datasets/your_set.txt"
```
Dataset has to contain more words than the training range (default = 3).
<br>
<br>

#### [Go here](http://www.jacobplaster.net/artificial-machine-learning-writer) to see results!
Here I show multiple text generations with different training data sets and how accurate the program is at impersonating the training data.
