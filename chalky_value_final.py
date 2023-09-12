import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
# image = cv2.imread('Images/img_chalky.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('Images/1.jpg', cv2.IMREAD_GRAYSCALE)

# Create a structuring element for the extended maxima transform
kernel = np.ones((3,3), np.uint8)  
# Apply Canny edge detector to preprocess the image
edges = cv2.Canny(image, 100, 200)
# Apply morphological reconstruction
reconstruction = cv2.dilate(edges, kernel, iterations = 1)
# Apply extended maxima transform
extended_maxima = reconstruction - edges
# Calculate the total area of chalky regions
chalky_area = np.sum(extended_maxima == 255)


print("Chalky Area:", chalky_area)

# Find contours in the extended maxima image
contours, _ = cv2.findContours(extended_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate the total area of rice seeds
total_rice_area = 0
for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    print(f"Area of Rice Seed {idx + 1}: {contour_area}")
    total_rice_area += contour_area

print("Total area of Rice Seeds:", total_rice_area)

chalky_percentage = (chalky_area / total_rice_area) *100
print("Chalky percentage: ", chalky_percentage)

if(chalky_percentage > 30):
    quality = "Bad Quality"
else:
    quality  = "Good Quality"

print(f"Percentage of chalky area: {chalky_percentage:.2f}%")
print(f"Quality: {quality}")

plt.subplot(141)
plt.imshow(edges, "gray")

plt.subplot(142)
plt.imshow(reconstruction, "gray")

plt.subplot(143)
plt.imshow(extended_maxima, "gray")

# plt.subplot(144)
# plt.imshow(chalky_area, "gray")

plt.show()


