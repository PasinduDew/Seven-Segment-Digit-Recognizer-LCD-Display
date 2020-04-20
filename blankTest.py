import cv2
import numpy
import csv

def isBlankDigit(image):
    flatList = image.flatten()
    print(flatList)
    noOfHighs = 0
    noOfLows = 0
    for val in flatList:
        if val > 250 :
            noOfHighs += 1
    print(noOfHighs / flatList.size)
    if noOfHighs / flatList.size > 0.95 :
        return True
    else : 
        return False

    




rawImage = cv2.imread("./imageSet/Raw/WIN_20200409_09_49_15_Pro.jpg")
# cv2.imshow("Raw Image", rawImage)

# rawImage = cv2.resize(rawImage, (1280, 960))

greyImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grey Image", greyImage)

# 1.2MP 1280x960
croppedImage = greyImage[325:648, 100:1198]
# 0.9MP 1280x720
# croppedImage = greyImage[205:525, 100:1198]

cv2.imshow("Cropped Image", croppedImage)

ret,threshImage = cv2.threshold(croppedImage, 30, 255, cv2. THRESH_BINARY)
cv2.imshow("Thresholded Image", threshImage)

print(threshImage.shape)
marginList = [(0, 215), (225, 440), (450, 655), (680, 880), (890, threshImage.shape[1] )]
print(marginList)

rows = []
index = 0
# Prefix of the Output Image File
outputFileNamePrefix = "testImg_"
# Extension of the Output Image File
outputFileExtention = "jpg"
dirTestImages = "./imageSet/testImages_3"
for i in range(5):
    print(marginList[i][0])
    digit = threshImage[:, marginList[i][0] : marginList[i][1]]
    digitText = "Digit " + str(i)
    cv2.imshow(digitText, digit)
    if isBlankDigit(digit) != True : 
        print("Blank Digit Found")
        digit = cv2.resize(digit, (28, 28))
        print(digit.shape)
        flatList = digit.flatten()
        newRow = [0]
        for val in flatList:
            newRow.append(val)
        rows.append(newRow)
        
        cv2.imwrite(dirTestImages +  "/" + outputFileNamePrefix + str(index) + "." + outputFileExtention, digit)
        index += 1


# digit_0 = threshImage[:, 0:215]
# isBlankDigit(digit_0)
# cv2.imshow("Digit 0", digit_0)

# digit_1 = threshImage[:, 225:440]
# cv2.imshow("Digit 1", digit_1)

# digit_2 = threshImage[:, 450:655]
# cv2.imshow("Digit 2", digit_2)

# digit_3 = threshImage[:, 680:880]
# cv2.imshow("Digit 3", digit_3)

# digit_4 = threshImage[:, 890:]
# cv2.imshow("Digit 4", digit_4)

# cv2.imwrite("digit_0.jpg", digit_0)
# cv2.imwrite("digit_1.jpg", digit_1)
# cv2.imwrite("digit_2.jpg", digit_2)
# cv2.imwrite("digit_3.jpg", digit_3)
# cv2.imwrite("digit_4.jpg", digit_4)



cv2.waitKey()

