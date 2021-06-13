import random
from collections import defaultdict
import sys
import os
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from math import sqrt,exp,pi
import numpy as np
import copy
import keyboard


user_input = list(sys.argv) # Taking input from terminal.
classifier_choice = 1 # 1 - Naive Bayes, 0 - k-nearest neighbor.
K_value = 0
for i in range(len(user_input)):
    if user_input[i] == "-k":
        classifier_choice = 0
        K_value = user_input[i+1]
    if user_input[i] == "-o":
        output_directory = user_input[i+1]


training_path = user_input[-2]

testing_path = user_input[-1]

# ----- TESTING DATA EXTRACTION ----------------------------------------------------------------------------------------

img_dict_t = dict() # {img_filename: pixel_data}
testing_data = []
for subdir,dirs,files in os.walk(testing_path):
    for images in files:
        if images.split(".")[1] == "dsv": # To avoid errors.
            continue
        img_t = Image.open(testing_path + "/" + images).convert("1") # Gives pixel values ranging from 0-255
        pixels = list(img_t.getdata())
        testing_data.append(pixels)
        img_dict_t[images] = pixels

# A dictionary with key being the filename and value being the pixel data has been generated.

#------------TRAINING DATA EXTRACTION-----------------------------------------------------------------------------------

# Launching the truth.dsv file and creating a dict for sorting the classes.

# Extracting the pixel data from the given directory. Gathering all the pixel data for each of the image file.

image_size = 0
img_dict = dict()
for subdir,dirs,files in os.walk(training_path):
    for images in files:
        if images.split(".")[1] == "dsv":
            continue
        img = Image.open( training_path + "/" + images).convert("1")
        pixels = list(img.getdata())
        width, height = img.size
        image_size = width
        img_dict[images] = pixels # {img_filename : [pixel data]}


training_guide = "/truth.dsv"

class_dict = {} # {img_filename : real value}

with open(training_path + training_guide) as f:
    truth_data = f.readlines()
    new_list = []
    for lines in truth_data:
        curr_line = lines.split(":")
        if curr_line[1][0] in class_dict:
            class_dict[curr_line[1][0]].append(curr_line[0])
        else:
            class_dict[curr_line[1][0]] = [curr_line[0]]

# Now we merge the pixel data as the "values" for each "key" (the class values).
training_dict = {} # {label: [pixel data of the image]}
for key_i in img_dict.keys():
    for key_c, value in class_dict.items():
        for imgs in value:
            if key_i == imgs:
                if key_c in training_dict.keys():
                    training_dict[key_c].append(img_dict[key_i])

                else:
                    training_dict[key_c] = [img_dict[key_i]]

train_data = [] # [label of the image, pixel data[0], pixel_data[1]....]
class_ids = set()
for key,value in training_dict.items():
    for val in value:
        train_data.append([key]+val)
        class_ids.add(key)


#-------NB ALGORITHMS--------------------------------------------------------------------------------------------------

    # First, helper functions are written down.
if classifier_choice == 1:
    def frequency_counter(dct, pixels, label):
        # Updates the dictionary containing the frequencies for each individual pixel.
        # [number of whites, number of blacks]
        for pixel in range(len(pixels)):
            color = pixels[pixel]
            dct[pixel][label][color] += 1

        return dct

    def norm_list(list):
        total = sum(list)
        for val in range(len(list)):
            list[val] /= total
        return list

    # Helper function
    def normalize_likelihood_dict(dct, img_size, labels):
        # Special function that normalizes the frequency values based on the given pixel training data.
        for pixel in range(img_size * img_size):
            for label in labels:
                dct[pixel][label] = norm_list(dct[pixel][label])
        return dct

    # Helper Function
    def normalize_dict(dct):
        # Special function that normalizes a given dictionary.
        total = sum(dct.values())
        if total != 0:
            for val in dct:
                dct[val] /= total
            return dct
        else:
            total = len(dct)
            for val in dct:
                dct[val] = 1/total
            return dct

    # Helper function that converts pixel intensities to binary values.
    # functions for training and testing data separated due to the presence of labels at the beginning.
    # pixel value converted to on/off. Pixel value > 0 (white) - 0 (off), Pixel value = 0 (black) - 1 (on)
    def binary_tran(data): # for training
        return [data[0]] + [0 if pixel > 64 else 1 for pixel in data[1:]]

    def binary_tran_t(data): # for testing
        return [0 if pixel > 64 else 1 for pixel in data]

    # Main training function.
    def training_NB():
        """
        Main training function that calculates the "prior" and "liklihood" values given a training dataset.

        :return: {Nested {dict}} containing frequency data for each individual label, for each individual pixel.
                 {label:probability(label)} Containing the probabilies for each label
        """
        prior = defaultdict(int) # Initialising a dict for storing the "prior" probabilities for all discovered labels.

        # Now a dict is to be created containing the frequency of pixels being black or white for each individual pixel.
        likelihood = {x: 0 for x in range(image_size * image_size)}

        # Now, to create lists for each pixel (which are the keys), the number of lists corresponding to the number of
        # classes present. Each list will have two values; signifying the frequency of the pixel being white or black.
        # [white, black].
        for k,v in likelihood.items():
            new_dict = {label: [0,0] for label in class_ids}
            likelihood[k] = new_dict

        # Now the dict will be updated with the frequencies after running through the training data and will be normalized
        # later to provide probabilities.
        for pixel_data in train_data:
            pixel_data = binary_tran(pixel_data)
            label = pixel_data[0]
            pixel_range = pixel_data[1:]
            prior[label] += 1 # Updates the frequency of class appearance.
            likelihood = frequency_counter(likelihood, pixel_range, label) # updates the color frequency for each pixel.

        prior = normalize_dict(prior) # Normalizing the values to range of 0 to 1.

        # Normalizing the color frequencies for each pixel value
        likelihood = normalize_likelihood_dict(likelihood, image_size, class_ids)

        return prior, likelihood

    def return_max_label(dict):
        # For given dict, outputs the best possible label by maxing on the probabilites (the values).
        max_val = dict[0]
        max_label = list(dict)[0]
        for label,val in dict.items():
            if val > max_val:
                max_label = label
                max_val = val
        return max_label


    # Main prediction function.
    def predict(data, prior_value, likelihood_values):
        """
        Main prediction function that outputs the prediction made. Utilizes the bayes theorm and calculates the
        probability of the label, given images as test data. The label with the highest probability is then outputted.

        :param data: [list] containing pixel data of the image to be predicted.
        :param prior_value: {dict} Containing the probility of each class.
        :param likelihood_values: {Nested{dict} Containing the probility of "feature given label" for each label.
        :return: str - the predicted label.
        """
        pixel_data = binary_tran_t(data) # Convert pixel data to binary for simpler classification.

        prior_copy = copy.deepcopy(prior_value)

        for pixel_id in range(len(data)):
            for label in class_ids:
                # Main Bayes algorithm; combining probabilities for a class given a feature.
                prior_copy[label] *= likelihood_values[pixel_id][label][pixel_data[pixel_id]]

            # Normalize the probabilities obtained so far.
            prior_copy = normalize_dict(prior_copy)

        return return_max_label(prior_copy) # return the class id with the highest probability.


    def run_test(prior_values, likelihood_dict):

        pixel_data = testing_data
        for pixel_id in range(len(pixel_data)):

            pixels = pixel_data[pixel_id]
            prediction = predict(pixels, prior_values, likelihood_dict)
            for image_filename, pixel_d in img_dict_t.items():
                if pixels == pixel_d:
                    img_dict_t[image_filename] = prediction # Replace the pixel data present with the prediction.


#-- MAIN - NB ----------------------------------------------------------------------------------------------------------

    prior, likelihood_dict = training_NB()
    run_test(prior,likelihood_dict)

    for key,value in img_dict_t.items():
        img = Image.open(testing_path + "/" + str(key))
        img.show()
        print("classification :", value)
        print("Press enter to continue..")
        keyboard.wait("enter")

#-----------k-NN ALGORITHM----------------------------------------------------------------------------------------------
if classifier_choice == 0:

    def get_dist_train(X_train, test_data):
        # Returns a list of "distances" from the sample datapoint to all the datapoints in the training dataset.
        return [calculate_distance(train_data, test_data) for train_data in X_train]

    def calculate_distance(x, y):
        # Euclidean distance formula
        return np.sqrt(sum([(x_a - y_b) ** 2  for x_a, y_b in zip(x,y)]))


    def most_frequent(lst):
        # Outputs the most frequently appearing value in a given list.
        known_values = {}
        for value in lst:
            if value in known_values.keys():
                known_values[value] += 1
            else:
                known_values[value] = 1
        return max(known_values, key=lambda k: known_values[k])

    def sort_according_to_distance(list):
        # Outputs a list, which consists of index values of the the original distance values that were sorted in an
        # ascending order.
        output_list = []
        list = sorted(enumerate(list), key=lambda x: x[1]) # Sorts the list according to the distance value.
        # And now we pluck out the index values which have been arranged.
        for dist_idx_pair in list:
            output_list.append(dist_idx_pair[0])

        return output_list

    def choose_candidates(label_list, sorted_distances, k):
        # As the list is sorted, cuts out the distances that are "greater than k".
        # outputs a list of candidates applicable for the given k value.
        output_list = []
        for values in sorted_distances[:int(k)]:
            output_list.append(label_list[values])

        return output_list



    def knn(train_dat, test_dat, k = K_value):
        """
        The main k- nearest neighbor algorithm function. Updates the img_dict_t dictionary with the prediction.


        :param train_dat: [Nested [list]] containing the training data.
        :param test_dat: [Nested [list]] containing the data to be predicted.
        :param k: {int} K value
        :return: Updates the img_dict_t
        """
        X_train = [] # Contains "flattened" pixel data for each image [img1_pixeldata, img2_pixeldata....]
        y_train = [] # Contains the true label for each image [img1_label, img2_label....]
        for pixel_id in range(len(train_dat)):
            X_train.append(train_dat[pixel_id][1:]) # the pixel data
            y_train.append(train_dat[pixel_id][0]) # the labels

        X_test = test_dat
        for sample in X_test:
            # First, we calculate 'distance' from the current sample till each of the images present in the training set.
            dist_for_train = get_dist_train(X_train,sample)

            # Then, we sort the list according to their distances.
            sorted_distances = sort_according_to_distance(dist_for_train)

            # Then, we select the applicable candidates based on our "k" value.
            candidates = choose_candidates(y_train,sorted_distances,k)

            # Finally, the most frequently occuring value is taken as the right guess and updated in the orignal dict.
            for img_name, pixel_data in img_dict_t.items():
                if pixel_data == sample:
                    img_dict_t[img_name] = most_frequent(candidates)


# --------MAIN - kNN --------------------------------------------------------------------------------------------------

    knn(train_data,testing_data)

    for key, value in img_dict_t.items():
        img = Image.open(testing_path + "/" + str(key))
        img.show()
        print("classification :", value)
        print("Press enter to continue..")
        keyboard.wait("enter")

