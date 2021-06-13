# Image-classifier
Image classifying program that utilizes Naive Bayes and Nearest Neighbors algorithms to classify a set of digit and alphabets.

## Datasets
I recommend the use fo your own dataset, perhaps the MNIST dataset, as the more training data is available to the algorithm, the better it's performance.
You may download a set of each, testing and training data, from this [link](https://drive.google.com/drive/folders/1E4UcSEuAquXD-riOIfxO5VYZbxjnFUwj?usp=sharing).  
NOTE : If you wish to use your own datasets, make sure there exists a truth file of type .dsv containing the real values for the training data in the following format 
- Name of the image : Real value of the image 
- Name of the image : Real value of the image
- .
- .
- .

## How to run
>> python3.8 classifier.py -h
* usage: classifier.py [-h] [-k K] [-b] train_path test_path

positional arguments:
  * train_path  Path to the training data
  * test_path   Path to the testing data

optional arguments:
  * -h, --help  show this help message and exit
  * -k K        run K-NN classifier
  * -b          run Naive Bayes classifier

## Output
Image files will launch using your default image viewer and the classification of the respective classifier will appear in the terminal.
Can iterate through the files using the "Enter" key.

## Required libraries

* PILLOW
* keyboard
* numpy

## Previews

![preview](https://www.mediafire.com/file/hh0prb5llvcwim4/1.PNG/file)
  
