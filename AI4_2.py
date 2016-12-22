# AI HW 4 Part 2


import math
import random
import numpy as np
import time


# Global variables
BIAS = 1
WEIGHTS = 0
EPOCHS = 1
ALPHA = 1000 / (1000 + EPOCHS)
ORDERING = 0                    # 0: fixed, 1: random

# Global constants
SIZE = 28
TRAIN_SIZE = 5000
TEST_SIZE = 1000


# Parse files
def parse_files(fileimage, filelabel):
    ''' Parses image and label files. Returns images_vec: vector of images,
        and labels_vec: vector of integer labels.
    '''
    file1 = open(fileimage, 'r')
    file2 = open(filelabel, 'r')
    images_vec = []
    labels_vec = []
    image = []
    y = 0

    for line in file1:
        image.append(line)
        y += 1
        if y == SIZE:
            y = 0
            images_vec.append(image)
            image = []

    for line in file2:
        labels_vec.append(int(line))

    file1.close()
    file2.close()

    return images_vec, labels_vec


# Decodes image
def decode_image(image):
    ''' Decodes image '''
    binary_image = []

    for y in range(SIZE):
        for x in range(SIZE):
            if image[y][x] == ' ':
                binary_image.append(0)
            elif image[y][x] == '#' or image[y][x] == '+':
                binary_image.append(1)

    return binary_image


# Training function
def training(weights_vec):
    ''' Training function '''
    # Parse files
    images_vec, labels_vec = parse_files('digitdata/trainingimages', 'digitdata/traininglabels')

    # Lists for counting accuracy
    label_correct = [0] * 10
    label_count = [0] * 10
    correct = 0

    # Define ordering of training data
    indices = list(range(TRAIN_SIZE))
    if ORDERING == 1:
        random.shuffle(indices)

    start = time.time()

    # Iterate through training data
    for idx in indices:

        image = images_vec[idx]
        label = labels_vec[idx]
        binary_image = decode_image(image)

        # Accumulate label count
        label_count[label] += 1
        
        # Initialize perceptron values with bias
        perceptrons = [BIAS] * 10

        # Calculate perceptron values for each class
        for i in range(10):
            for y in range(SIZE):
                for x in range(SIZE):
                    index = x + y*SIZE
                    perceptrons[i] += weights_vec[i][index] * binary_image[index]

        # Classify by finding maximum perceptron value
        maximum = max(perceptrons)
        label_predict = perceptrons.index(maximum)

        # Accumulate accuracy counts
        if label_predict == label:
            label_correct[label] += 1
            correct += 1
        # If classification fails, update weights for predicted and actual classes
        else:
            for y in range(SIZE):
                for x in range(SIZE):
                    index = x + y*SIZE
                    weights_vec[label][index] += ALPHA * binary_image[index]
                    weights_vec[label_predict][index] -= ALPHA * binary_image[index]

    end = time.time()
    print('Time:', end - start)

    # Calculate accuracy
    training_accuracy = correct / TRAIN_SIZE

    return weights_vec, training_accuracy


# Testing function
def testing(weights_vec):
    ''' Testing function '''
    # Parse files
    images_vec, labels_vec = parse_files('digitdata/testimages', 'digitdata/testlabels')

    # Lists for counting accuracy
    label_correct = [0] * 10
    label_count = [0] * 10
    correct = 0

    confusion_matrix = np.zeros((10,10))

    # Iterate through training data
    for idx in range(TEST_SIZE):
        image = images_vec[idx]
        label = labels_vec[idx]
        binary_image = decode_image(image)

        # Accumulate label count
        label_count[label] += 1
        
        # Initialize perceptron values with bias
        perceptrons = [BIAS] * 10

        # Calculate perceptron values for each class
        for i in range(10):
            for y in range(SIZE):
                for x in range(SIZE):
                    index = x + y*SIZE
                    perceptrons[i] += weights_vec[i][index] * binary_image[index]

        # Classify by finding maximum perceptron value
        maximum = max(perceptrons)
        label_predict = perceptrons.index(maximum)

        # Update confusion matrix
        confusion_matrix[label][label_predict] += 1

        # Accumulate accuracy counts
        if label_predict == label:
            label_correct[label] += 1
            correct += 1

    # Normalize confusion matrix
    for i in range(10):
        for j in range(10):
            confusion_matrix[i][j] /= label_count[i]

    # Calculate accuracies
    label_accuracy = []
    for i in range(10):
        label_accuracy.append(label_correct[i] / label_count[i])

    testing_accuracy = correct / TEST_SIZE

    return testing_accuracy, label_accuracy, confusion_matrix


# Main function
def main():
    ''' Main function '''
    # Initialize weights vector
    weights_vec = []
    for i in range(10):
        weights_vec.append([WEIGHTS] * SIZE * SIZE)

    # Perform training
    training_curve = []
    for i in range(EPOCHS):
        weights_vec, training_accuracy = training(weights_vec)
        training_curve.append(training_accuracy)

    # Perform testing
    testing_accuracy, label_accuracy, confusion_matrix = testing(weights_vec)

    # Print results
    print('Training Curve:', training_curve)
    print('Testing Accuracy:', testing_accuracy)
    print('\nLabel Accuracy:')
    for i in range(10):
        print(i, ':', label_accuracy[i])
    print('\nConfusion Matrix:', confusion_matrix)


# Main function call
if __name__ == '__main__':
    main()