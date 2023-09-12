import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
# image = cv2.imread('all_imp.jpg')
image = cv2.imread("imp_1.jpg", 1)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of colors for potential impurities in HSV
lower_color = np.array([0, 50, 50])  
upper_color = np.array([20, 255, 255])  
# Create a mask to isolate potential impurities
color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

# Find contours in the mask
contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total area of impurities in pixels
total_impurity_area = 0
for contour in contours:
    total_impurity_area += cv2.contourArea(contour)

# Calculate the total area of the rice image
total_rice_area = image.shape[0] * image.shape[1]

# Calculate the percentage of impurity area
impurity_percentage = (total_impurity_area / total_rice_area) * 100

# Draw contours on the original image
rice_with_contours = image_rgb.copy()
cv2.drawContours(rice_with_contours, contours, -1, (0, 0, 255), 2)

# Display the original image with contours
# cv2.imshow('Impurity Detection', rice_with_contours)

# Print the impurity percentage
print(f"Percentage of impurity area: {impurity_percentage:.2f}%")



plt.subplot(1,2,1)
plt.axis("off")
plt.title("Rice with impurity ", color ="maroon", fontsize = 12)
plt.imshow(image_rgb)

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Impurity Detection ", color ="maroon", fontsize = 12)
plt.imshow(rice_with_contours)

plt.show()