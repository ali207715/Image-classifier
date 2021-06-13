Image-classifier
Image classifying program that utilizes Naive Bayes and Nearest Neighbors algorithms to classify a set of digit and alphabets.

---How to run---
>> python3.8 classifier.py -h
usage: classifier.py [-h] [-k K] [-b] train_path test_path

positional arguments:
  train_path  Path to the training data
  test_path   Path to the testing data

optional arguments:
  -h, --help  show this help message and exit
  -k K        run K-NN classifier
  -b          run Naive Bayes classifier
  -o name     name (with path) of the output dsv file with the results
  
  
