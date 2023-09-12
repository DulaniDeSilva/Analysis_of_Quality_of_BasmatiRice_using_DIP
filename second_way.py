import cv2
import numpy as np
import matplotlib.pyplot as plt

def classification(ratio):
    claz = ""
    if ratio >= 5:
        claz = "Very Long Grain"
    elif(ratio < 5 and ratio >= 4):
        claz = "Long Grain"
    elif(ratio < 4 and ratio >= 3):
        claz = "Medium Size Grain"
    elif(ratio <3 and ratio >= 2):
        claz = "Short Grain"
    else:
        claz = "Broken Grain"
    return claz

#loading the image in grayscale
img = cv2.imread("Images/img_01_pure.jpg", 0)

#convertion into binary
ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((5,5), np.float32)/9
dst = cv2.filter2D(thresh, -1, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion
erosion = cv2.erode(dst, kernel2, iterations=1)

#dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)

#edge detection
edges = cv2.Canny(dilation, 100, 200)
edges_copy = edges.copy()

#size detection
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_seeds = len(contours)
print("No of rice grains = ", len(contours))
total_ar = 0

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h
    if(aspect_ratio<1):
        aspect_ratio=1/aspect_ratio
    print( round(aspect_ratio,2),classification(aspect_ratio))
    total_ar+=aspect_ratio
    
avg_ar=total_ar/len(contours)
print("Average Aspect Ratio=",round(avg_ar,2),classification(avg_ar))





plt.subplot(2,3,1)
plt.axis("off")
plt.title("Original Image")
plt.imshow(img, "gray")

plt.subplot(2,3,2)
plt.axis("off")
plt.title("Binary Image")
plt.imshow(thresh, "gray")

plt.subplot(2,3,3)
plt.axis("off")
plt.title("Eroded Image")
plt.imshow(erosion, "gray")

plt.subplot(2,3,4)
plt.axis("off")
plt.title("Dilated Image")
plt.imshow(thresh, "gray")

plt.subplot(2,3,5)
plt.axis("off")
plt.title("edge detected Image")
plt.imshow(edges, "gray")

plt.subplot(2,3,6)
plt.axis("off")
plt.title("Labeled Count")
plt.imshow(edges_copy, "gray")



plt.show()