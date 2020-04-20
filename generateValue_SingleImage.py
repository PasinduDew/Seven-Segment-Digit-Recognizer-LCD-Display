import cv2
import numpy
import os
import csv
from NeuralNetwork import neuralNetwork
from captureImage import captureImage



path = 'F:\\Electronics\\MyProjects\\7-Segment Recognizer\\ImageSet\\RawTest_4'


# --------------------------------------------------------------------------------------------------------------------
#                                         Function Definitions
# --------------------------------------------------------------------------------------------------------------------

# To Check Whether the Area which is cropped is a blank area without a digit
def isBlankDigit(image):
    flatList = image.flatten()
    # print(flatList)
    noOfHighs = 0
    noOfLows = 0
    for val in flatList:
        if val > 250 :
            noOfHighs += 1
    # print(noOfHighs / flatList.size)
    if noOfHighs / flatList.size > 0.95 :
        return True
    else : 
        return False



# --------------------------------------------------------------------------------------------------------------------
#                                         Neural Network Configuration
# --------------------------------------------------------------------------------------------------------------------

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.25

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("seven_segment_dataset_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# ------------------------------------ train the neural network ----------------------------------
# go through all records in the training data set
# epochs is the number of times the training data set is used for training


# ------------------------------------- Train From a New Dataset ------------------------------------
"""
epochs = 10
for i in range(epochs) :
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        # print(targets)
        n.train(inputs, targets)
        pass
    pass

# n.saveWeights("weightValuesFullDataset")
"""
# ------------------------------------- Load Weights From the File ------------------------------------
n.loadWeights("weightValuesFullDataset.csv")



# -------------------------------------------------------------------------------------------------------------------------------------
#                                    Creating the Output Image Files and CSV - Dataset Creation
# -------------------------------------------------------------------------------------------------------------------------------------
# fileName = "lcdImage.jpg"
rawImage = captureImage()
  
# rawImage = cv2.imread(fileName)
# cv2.imshow(fileName, rawImage)

greyImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grey Image", greyImage)

# Cropping the Image to extract the Area Where the Digigit Appear
# 1.2MP 1280x960
croppedImage = greyImage[325:648, 100:1198]
# 0.9MP 1280x720
# croppedImage = greyImage[205:525, 100:1198]
# cv2.imshow("Cropped Image", croppedImage)

# Thresholding the Image
ret,threshImage = cv2.threshold(croppedImage, 50, 255, cv2. THRESH_BINARY)
# Margings/ Bounderies of the Digits -> For Cropping
marginList = [(0, 215), (225, 440), (450, 655), (680, 880), (890, threshImage.shape[1] )]

valStr = ""

# There are only 5 digits to extracted from the LCD Display
for i in range(5):
    # print(marginList[i][0])
    digit = threshImage[:, marginList[i][0] : marginList[i][1]]

    # digitText = "Digit " + str(i)
    # cv2.imshow(digitText, digit)

    if isBlankDigit(digit) != True : 
        # print("Blank Digit Found")
        digit = cv2.resize(digit, (28, 28))
        # print(digit.shape)
        flatList = digit.flatten()
        
        
        # scale and shift the inputs
        inputs = (numpy.asfarray(flatList) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        valStr += str(label)
        # print("Network's Value: ", label)

    if i == 2 : 
        valStr += "."

print("Value String: ", valStr)

