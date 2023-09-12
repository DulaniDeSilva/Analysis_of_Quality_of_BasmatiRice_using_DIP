import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'

#loading the image 
img_bgr = cv2.imread("Images/img_01_pure.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

median_filtered = cv2.medianBlur(img, 5)  # Adjust the kernel size (odd number) as needed

#convertion into binary
ret, thresh = cv2.threshold(median_filtered, 160, 255, cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((3,3), np.float32)/9
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
print("No of rice grains = ", num_seeds)
contour_draw = cv2.drawContours(dilation_color, contours, -1, (0,255,0), 2)

#initialize list to store axis lengths
major_axis_length = []
minor_axis_length = []

#ratio list
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

        #putting the text 
        cv2.putText(contour_draw, str(seed_number), (int(center[0]), int(center[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)





plt.subplot(2,3,1)
plt.axis("off")
plt.title("Original Image", color = COLOR, fontsize = FONT_SIZE)
plt.imshow(img_rgb)

plt.subplot(2,3,2)
plt.axis("off")
plt.title("Binary Image",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(thresh, "gray")

plt.subplot(2,3,3)
plt.axis("off")
plt.title("Eroded Image",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(erosion, "gray")

plt.subplot(2,3,4)
plt.axis("off")
plt.title("Dilated Image",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(thresh, "gray")

plt.subplot(2,3,5)
plt.axis("off")
plt.title("edge detected Image",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(edges, "gray")


plt.subplot(2,3,6)
plt.axis("off")
plt.title("Total Rice Count",color = COLOR, fontsize = FONT_SIZE)
plt.imshow(contour_draw)


plt.show()