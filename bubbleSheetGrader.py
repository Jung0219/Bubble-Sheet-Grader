import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def sortContours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv.boundingRect(c) for c in cnts]
    pair = zip(*sorted(zip(cnts, boundingBoxes),
               key=lambda b: b[1][i], reverse=reverse))
    return pair


def labelContours(img, cnts, number):
    M = cv.moments(cnts)
    cX = int(M["m10"]/M["m00"])
    cY = int(M["m01"]/M["m00"])

    cv.putText(img, "{}".format(number), (cX, cY),
               cv.FONT_HERSHEY_COMPLEX, fontScale=3, color=(0, 0, 255), thickness=10)

    # x, y, w, h = cv.boundingRect(cnts)
    # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)


def isCircle(contour):
    x, y, w, h = cv.boundingRect(contour)
    ratio = w / h

    if w > 180 and h > 180 and ratio > 0.9 and ratio < 1.1:
        return True
    return False


def alphabetToNumber(alphabet):
    if alphabet == "A":
        return 0
    if alphabet == "B":
        return 1
    if alphabet == "C":
        return 2
    if alphabet == "D":
        return 3
    if alphabet == "E":
        return 4


answers = {1: "B", 2: "C", 3: "A", 4: "A", 5: "D",
           6: "B", 7: "D", 8: "D", 9: "B", 10: "E",
           11: "A", 12: "C", 13: "C", 14: "D", 15: "C",
           16: "B", 17: "E", 18: "A", 19: "A", 20: "D"}

for question in answers.keys():
    answers[question] = alphabetToNumber(answers[question])

# preprocessing
# --------------------------------------------------------------------------------------------------------------------------------------

address = r"C:\Users\OWNER\Downloads\python\OpenCV\images\OMR\omr.jpg"
img = cv.imread(address)

# to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# to B/W
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# binarize
thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)[1]

# fill out the holes
M = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, M, iterations=1)

# find contours, and the largest one
contours = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
large = max(contours, key=cv.contourArea)

# find the bounding rectangle of the sheet
rect = cv.minAreaRect(large)
box = cv.boxPoints(rect)

# make it into integer coordinates
box = np.intp(box)

botLeft = box[0]
topLeft = box[1]
topRight = box[2]
botRight = box[3]

extremePoints = np.array(
    [topLeft, topRight, botRight, botLeft], dtype=np.float32)

# calculate the new width using pythagorean theorem
x1 = np.diff([topLeft[0], topRight[0]])
y1 = np.diff([topLeft[1], topRight[1]])
x2 = np.diff([botLeft[0], botRight[0]])
y2 = np.diff([botLeft[1], botRight[1]])

w = int(min(np.sqrt(x1 ** 2 + y1 ** 2)[0], np.sqrt(x2 ** 2 + y2 ** 2)[0]))

x1 = np.diff([topRight[0], botRight[0]])
y1 = np.diff([topRight[1], botRight[1]])
x2 = np.diff([topLeft[0], botLeft[0]])
y2 = np.diff([topLeft[1], botLeft[1]])

h = int(min(np.sqrt(x1 ** 2 + y1 ** 2)[0], np.sqrt(x2 ** 2 + y2 ** 2)[0]))

# destination size
dst = np.array([[0, 0], [w-3, 0], [w-3, h-3], [0, h-3]], dtype="float32")

# obtain bird-eyes view
M = cv.getPerspectiveTransform(extremePoints, dst)
warpedOpen = cv.warpPerspective(opened, M, (w, h))
warpedImg = cv.warpPerspective(img, M, (w, h))

# sorting bubble contours to a usable format
# ---------------------------------------------------------------------------------------------------------------------------------------

# find circles among contours
cnts = cv.findContours(warpedOpen, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
circles = []
for contour in cnts:
    if isCircle(contour):
        circles.append(contour)

firstTenQuestions = []
secondTenQuestions = []

# divide circles into two groups based on x position (1-10 and 10-20)
for circle in circles:
    x, y, w, h = cv.boundingRect(circle)
    secondTenQuestions.append(
        circle) if x > 2250 else firstTenQuestions.append(circle)

# sort them from top to bottom
(firstTenQuestions, _) = sortContours(firstTenQuestions, "top-to-bottom")
(secondTenQuestions, _) = sortContours(secondTenQuestions, "top-to-bottom")

allQuestions = firstTenQuestions + secondTenQuestions
omrSheet = {}

# sort them left to right, store it in dictionary
for i in np.arange(5, 101, 5):
    # by five
    row = allQuestions[i - 5:i]

    # omrSheet[questionNumber][a, b, c, d, e]
    (sortedRow, _) = sortContours(row)
    omrSheet.update({i//5: sortedRow})

# grading portion
# ----------------------------------------------------------------------------------------------------------------------------------------
correct = 0

# iterate over each answers
for quesNum, answer in answers.items():
    mask = np.zeros(warpedOpen.shape, dtype="uint8")
    cv.drawContours(mask, [omrSheet[quesNum][answer]], -1, 255, -1)
    answerRegion = cv.bitwise_and(warpedOpen, mask)

    # differentiate bubbles by nonzero pixels
    count = cv.countNonZero(answerRegion)
    if count < 1000:
        cv.drawContours(
            warpedImg, omrSheet[quesNum][answer], -1, (0, 255, 0), 15)
        correct += 1
    else:
        cv.drawContours(
            warpedImg, omrSheet[quesNum][answer], -1, (255, 0, 0), 15)

percentage = correct * 5

cv.putText(warpedImg, "{}% correct".format(percentage),
           (1500, 200), cv.FONT_HERSHEY_SIMPLEX, 5, 1, 10)

"""
plt.figure(figsize=[15, 15])
plt.imshow(warpedImg)
plt.title("{}%".format(percentage))
plt.waitforbuttonpress()
plt.close("all")
"""
