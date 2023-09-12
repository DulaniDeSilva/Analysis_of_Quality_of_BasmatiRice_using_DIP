# Importing of the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'

# Loading the images using cv2.imread() function
img_bgr = cv2.imread("Images/image_01.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Applying median filtering
median_filtered = cv2.medianBlur(img, 5)
# Applying threshold algorithm to segement rice grains from the black background
ret, thresh = cv2.threshold(median_filtered, 160, 255, cv2.THRESH_BINARY)


# loading the images using cv2.imread() function 
img_bgr = cv2.imread("Images/img_01_pure.jpg", 1)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# Applying median filtering
median_filtered = cv2.medianBlur(img, 5)  


# Applying threshold algorithm to segment rice grains from black background
ret, thresh = cv2.threshold(median_filtered, 160, 255, cv2.THRESH_BINARY)
#averaging filter
kernel = np.ones((3,3), np.float32)/9
dst = cv2.filter2D(thresh, -1, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


# Applying erosion to seperate the touching features of rice grains
erosion = cv2.erode(dst, kernel2, iterations=3)
# Apply dilation to grow the eroded features back to their original shape
dilation = cv2.dilate(erosion, kernel2, iterations=1)

# Apply edge detection algorithm  to find the region of boundaries of rice grains
edges = cv2.Canny(dilation, 100, 200)


# object Measurement
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_seeds = len(contours)

#Calculating major axis, minor axis and calculating ratio
for idx,contour in enumerate(contours):
    seed_number = num_seeds - idx
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h

    major_axis = w
    minor_axis = h






print("No of rice grains = ", num_seeds)
contour_draw = cv2.drawContours(dilation, contours, -1, (0,255,0), 2)

#initialize list to store axis lengths
major_axis_length = []
minor_axis_length = []
Aspect_ratio_of_seeds = []

#printing major axis, minor axis and calculating ratio
for idx, contour in enumerate(contours):
    if len(contour) >=5:
        seed_number = num_seeds - idx
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        aspect_ratio = float(major_axis) / minor_axis
        Aspect_ratio_of_seeds.append((seed_number, aspect_ratio))
       
        print("Seed Number: ", seed_number, " | ", 
              "Major axis length: ", major_axis, "|", 
              "Minor axis length: ", minor_axis, "|", 
              "Aspect Ratio: ",aspect_ratio, "|") 
              
        major_axis_length.append(major_axis)
        minor_axis_length.append(minor_axis)

        cv2.putText(contour_draw, str(seed_number), (int(center[0]), int(center[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



