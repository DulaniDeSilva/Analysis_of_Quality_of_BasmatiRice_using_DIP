import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'


#loading the image in grayscale
img_bgr = cv2.imread("Images/6.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#convertion into binary
ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((5,5), np.float32)/9
dst = cv2.filter2D(thresh, -1, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion
erosion = cv2.erode(dst, kernel2, iterations=3)

#dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)
dilation_color = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

#edge detection
edges = cv2.Canny(dilation, 100, 200)
edges_copy = edges.copy()

#size detection
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_seeds = len(contours)

#initialize list to store axis lengths
major_axis_length = []
minor_axis_length = []
major_axis_length_mm_list = []


#printing major axis, minor axis and calculating ratio
for idx,contour in enumerate(contours):
    seed_number = num_seeds - idx
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if(aspect_ratio < 1):
        aspect_ratio = 1/ aspect_ratio
    print(aspect_ratio)

    major_axis = w
    minor_axis = h

  
    major_axis_length.append(major_axis)
    minor_axis_length.append(minor_axis)
    conversion_factor = 415/6
    major_axis_length_mm = (major_axis )/ conversion_factor
    major_axis_length_mm_list.append(major_axis_length_mm)

long_slender_count , short_slender_count, medium_slender_count, long_bold_count, short_bold_count, broken_seed_count = 0,0,0,0,0,0


def categorise(major_axis_length, aspect_ratio):
    global long_slender_count, short_slender_count, medium_slender_count,long_bold_count,short_bold_count
    category = ""
    if(major_axis_length > 6 and aspect_ratio > 3):
        category = "Long Slender"
        long_bold_count += 1
        print("Count long Slender: ", long_slender_count)
    elif(major_axis_length <6 and aspect_ratio > 3):
        category = "Short Slender"
        short_slender_count += 1
        print("Count short Slender: ", short_slender_count)
    elif(major_axis_length < 6 and ((aspect_ratio > 2.5)and (aspect_ratio < 3))):
        category = "Medium Slender"
        medium_slender_count +=1
        print("Count medium slender:", medium_slender_count)
    elif(major_axis_length > 6 and aspect_ratio < 3):
        category = "Long Bold"
        long_bold_count += 1
        print("Long Bold:", long_bold_count)
    elif(major_axis_length < 6 and aspect_ratio < 2.5):
        category = "Short Bold"
        short_bold_count += 1
        print("Short Bold:", short_bold_count)
    else:
        category = "Broken_seeds"
        broken_seed_count += 1
        print("Broken Seed: ", broken_seed_count)
    return category


for idx, contour in enumerate(contour):
    rice_class = categorise(major_axis_length_mm, aspect_ratio)














plt.subplot(2,3,1)
plt.axis("off")
plt.title("Original Image", color = COLOR, fontsize = FONT_SIZE)
plt.imshow(img_rgb)

plt.subplot(2,3,2)
plt.axis("off")
plt.title("Quality based on length",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(dilation_color)

#Major axis distribution
plt.subplot(2,3,3)
plt.hist(major_axis_length, bins=100, color='green', alpha =0.7, edgecolor = 'black')
plt.xlabel("Major Axis Length")
plt.ylabel('frequency')
plt.title("Distribution of major axis lengths", color = "maroon",fontsize = 8)
plt.grid(True)

plt.subplot(2,3,4)
#Major axis distribution
plt.hist(minor_axis_length, bins=100, color='blue', alpha =0.7,edgecolor = 'black')
plt.xlabel("Minor Axis Length")
plt.ylabel('frequency')
plt.title("Distribution of minor axis lengths", color = "maroon", fontsize = 8)
plt.grid(True)

plt.show()

