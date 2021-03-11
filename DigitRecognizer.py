import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

#The Model it is trained from refer to (model.py)
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 512) ##fc1 means first fully connected layer
    self.fc2 = nn.Linear(512, 512)
    #self.fc3 = nn.Linear(64, 64)
    self.fc4 = nn.Linear(512, 10)

  #how the data will flow through the network
  def forward(self, x):
    x = x.view(-1, 28*28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    #x = F.relu(self.fc3(x))
    x = (self.fc4(x))
    return F.log_softmax(x, dim=1)


# Load the trained model
PATH = "TrainedModel.pt"
model = torch.load(PATH)
model.eval()

# Read the input image
im = cv2.imread("Images/IMG_3739-1.jpg")
cv2.imshow("img", im)

# Convert to grayscale and apply Gaussian Blur
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

#Create an empty array for the predictions
nums = []
# For each rectangular region, calculate HOG features and predict the digit using Linear SVM.
for rect in rects:
    # To draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make each rectangular region around the digit
    region = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - region // 2)
    pt2 = int(rect[0] + rect[2] // 2 - region // 2)
    roi = im_th[pt1:pt1+region, pt2:pt2+region]

    # Resizing the image to feed into the neural network
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    #prediction for each rectangle (on iteration)
    prediction = (torch.argmax(model(torch.from_numpy(roi).float()))).item()
    cv2.putText(im, str(prediction), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    nums.append(prediction)

#Create empty arrays for each of the vectors
v1 = []
v2 = []

#append each of the predictions to the end of the rectangle tuple
pog = [*zip(*zip(*rects), nums)]


#Sorting the array with all the positions of the rectangles to figure out which rectangle is which number
pog = (sorted(pog, key=lambda x: x[0], reverse=False))

for i in range(3):
    v1.append(pog[i])
for i in range(3,6):
    v2.append((pog[i]))

v1 = (sorted(v1, key=lambda x: x[1], reverse=False))
v2 = (sorted(v2, key=lambda x: x[1], reverse=False))

vec1 = []
vec2 = []

#format in vector form
for i in range(len(v1)):
    vec1.append(v1[i][4])
    vec2.append(v2[i][4])

print("The first vector is: ", vec1)
print("The second vector is: ", vec2)

#calculating dot product
dot = 0
for i in range(len(v1)):
    dot += (vec1[i] * vec2[i])

#calculating cross product
cross = 0
cross = [vec1[1]*vec2[2] - vec1[2]*vec2[1], vec1[2]*vec2[0] - vec1[0]*vec2[2], vec1[0]*vec2[1] - vec1[1]*vec2[0]]

# Display image with output text
print("The cross product of the vectors is: ", cross)
print("The dot product is: ", dot)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

