import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
# image = cv2.imread('Images/img_01_norm.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('Images/5.jpg', cv2.IMREAD_GRAYSCALE)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
# image = cv2.imread('Images/1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply thresholding to segment chalky areas
_, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Calculate total number of pixels in the image
# # total_pixels = image.shape[0] * image.shape[1]
# total_area = 0
# for idx, contour in enumerate(contours):
#     contour_area = cv2.contourArea(contour)
#     print(f"Area of Rice Seed {idx + 1}: {contour_area}")
#     total_area += contour_area
# print("Total area of Rice Seeds: ", total_area)


# # Calculate the total area of chalky regions in pixels
# chalky_pixels = 0
# for contour in contours:
#     chalky_pixels += cv2.contourArea(contour)
# print("Total chalky area: ", chalky_pixels)

# chalky_percentage = (chalky_pixels / total_area) *100
# print("Chalky percentage: ", chalky_percentage)



# Draw contours on the original image
rice_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(rice_with_contours, contours, -1, (0, 0, 255), 2)



# print(f"Percentage of chalky area: {chalky_percentage:.2f}%")
# print(f"Quality: {quality}")


# Display the original image with contours
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Rice Sample ", color ="maroon", fontsize = 12)
plt.imshow(image_rgb)

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Chalky Detection ", color ="maroon", fontsize = 12)
plt.imshow(rice_with_contours)
plt.show()
