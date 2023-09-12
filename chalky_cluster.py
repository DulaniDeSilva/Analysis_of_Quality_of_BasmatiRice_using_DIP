import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Images/img_chalky.jpg', cv2.IMREAD_GRAYSCALE)
image_copy = image.copy()

# Apply thresholding
ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
con1 = cv2.drawContours(image, contours, -1, (0,255,0), 2)
contoursr, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
con2 = cv2.drawContours(image_copy, contoursr, -1, (255, 0, 0), 2)

# Calculate the total area of rice seeds
total_rice_area = 0

# Calculate Chalky Area
# chalky_area = np.sum(thresh == 255)
total_rice_area = 0
total_chalky = 0
for idx, contour in enumerate(contoursr):
    contour_area = cv2.contourArea(contour)
    print(f"Area of Rice Seed {idx + 1}: {contour_area}")
    total_rice_area += contour_area

for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    print(f"Area of  {idx + 1}: {contour_area}")
    total_chalky += contour_area

print("Total area of Rice Seeds:", total_rice_area)
print("Total chalky area: ", total_chalky)

chalky_percentage = (total_chalky/ total_rice_area) *100
print("Chalky percentage: ", chalky_percentage)

if(chalky_percentage > 30):
    quality = "Bad Quality"
else:
    quality  = "Good Quality"

print(f"Percentage of chalky area: {chalky_percentage:.2f}%")
print(f"Quality: {quality}")






plt.subplot(131)
plt.imshow(con1, "gray")

plt.subplot(132)
plt.imshow(con2, "gray")

plt.show()
