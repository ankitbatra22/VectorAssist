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
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

PATH = "entire_model.pt"
model = torch.load(PATH)
model.eval()

# Read the input image
im = cv2.imread("IMG_3739-1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#print(im_th.shape)

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

    # Resizing the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    prediction = (torch.argmax(model(torch.from_numpy(roi).float()))).item()
    cv2.putText(im, str(prediction), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
#   print(prediction)
    nums.append(prediction)

#Create empty arrays for the vectors
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
print("The cross product of the vectos is: ", cross)
print("The dot product is: ", dot)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

