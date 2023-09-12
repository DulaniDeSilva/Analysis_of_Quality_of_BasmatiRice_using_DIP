import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'
# img_bgr = cv2.imread("Images/img_01_pure.jpg", 1)
# img_bgr = cv2.imread("Images/7.jpg", 1)
# img_bgr = cv2.imread("image06.jpg", 1)
img_bgr = cv2.imread("imp_2.jpg", 1) #no impurity
# img_bgr = cv2.imread("img1.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
# image = cv2.imread('Images/img_01_norm.jpg', cv2.IMREAD_GRAYSCALE)
image = img.copy()

median_filtered = cv2.medianBlur(img, 5)  # Adjust the kernel size (odd number) as needed

#convertion into binary

ret, thresh = cv2.threshold(median_filtered, 160, 255, cv2.THRESH_BINARY)


# dst = cv2.filter2D(thresh, -1, kernel)
#averaging filter
# kernel = np.ones((3,3), np.float32)/9



kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
#erosion
erosion = cv2.erode(thresh, kernel2, iterations=3)
#dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)




dilation_color = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)


dilation_color_2 = dilation_color.copy()
#edge detection
edges = cv2.Canny(dilation, 100, 200)
edges_copy = edges.copy()

#size detection
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num_seeds = len(contours)
print("No of rice grains = ", num_seeds)
# contour_draw = cv2.drawContours(dilation_color, contours, -1, (0,255,0), 2)
contour_draw_count = cv2.drawContours(dilation_color, contours, -1, (0,255,0), 2)

# based on the length of the grains
#initialize list to store axis lengths
major_axis_length = []
minor_axis_length = []

count_extra_long = 0
count_long = 0
count_medium = 0 
count_short = 0
count_broken = 0

count_slender = 0
count_medium = 0
count_bold = 0
count_round = 0


rice_varieties = {
    "Extra Long Grain": count_extra_long,
    "Long Grain": count_long,
    "Medium Grain": count_medium,
    "Short Grain": count_short,
    "Broken Grain": count_broken
}

shape_varieties = {
    "Extra Long Grain": count_slender,
    "Medium Grain": count_medium,
    "Short Grain": count_bold,
    "Broken Grain": count_round
}

def classification_length(axis_length):
    global count_extra_long, count_long, count_medium, count_short, count_broken
    length_class = ""
    if axis_length >= 85:
        length_class = "Extra Long Grain"
        count_extra_long += 1
        # print("count_extra_long:", count_extra_long)
    elif(axis_length < 85 and axis_length >= 60):
        length_class = "Long Grain"
        count_long += 1
        # print("count_long:", count_long)
    elif(axis_length < 60 and axis_length >= 50):
        length_class = "Medium  Grain"
        count_medium += 1
        # print("count_medium:", count_medium)
    elif (axis_length < 50 and axis_length >= 30):
        length_class = "Short Grain"
        count_short += 1
        # print("count_short:", count_short)
    else:
        length_class = "Broken Grain"
        count_broken += 1
        # print("count_broken:", count_broken)

    return length_class


def classification_ratio(ratio):
    global count_slender, count_medium, count_bold, count_round 
    ratio_class = ""
    if ratio >= 3:
        ratio_class = "Slender Grain Shape"
        count_slender += 1
        # print("Slender Shape count: ", count_slender)
    elif(ratio < 3 and ratio >= 2.1):
        ratio_class = "Medium Grain Shape"
        count_medium += 1
        # print("Medium Shape count: ", count_medium)
    elif(ratio < 2.1 and ratio >= 1.1):
        ratio_class = "Bold Size Grain"
        count_bold += 1
        # print("Bold Shape count: ", count_bold)
    else:
        ratio_class = "Round Grain"
        count_round += 1
        # print("Slender Shape count: ", count_round)
    return ratio_class

for idx, contour in enumerate(contours):
    if len(contour) >=5:
        seed_number = num_seeds - idx
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
       
        #putting the text 
        cv2.putText(contour_draw_count, str(seed_number), (int(center[0]), int(center[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)




#calculating  major axis, minor axis and calculating ratio
for idx,contour in enumerate(contours):
    seed_number = num_seeds - idx
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w)/h
    if(aspect_ratio < 1):
        aspect_ratio = 1/ aspect_ratio
    # print(aspect_ratio)

    major_axis = w
    minor_axis = h

    print("Seed Number: ", seed_number, " | ", 
              "Major axis length: ", major_axis, "|", 
              "Minor axis length: ", minor_axis, "|", 
              "Accoridng to length: ", classification_length(major_axis), "|",
              "Classification According to ratio: ", classification_ratio(aspect_ratio))
        
    major_axis_length.append(major_axis)
    minor_axis_length.append(minor_axis)


# print("********************Counts ********************")

# # Print the counts of each rice variety
# print("No of rice grains = ", len(contours))
# for variety, count in rice_varieties.items():
#     print(f"Count of {variety}: {count}")

total_seeds = len(contours)
percentage_extra_long_grain = (count_extra_long / total_seeds) * 100
percentage_long_grain = (count_long / total_seeds) * 100
percentage_medium_grain = (count_medium / total_seeds) * 100
percentage_short_grain = (count_short / total_seeds) * 100
percentage_broken_grain = (count_broken / total_seeds) * 100
print("No of rice grains = ", len(contours))
print()
# Print  the analysis
print("**********Analysis based on the major axis length of the rice granules**********")
print()
# # Display the analysis based on major axis length
# print("{:.2f}% of Very High Quality Rice".format(percentage_extra_long_grain))
# print("{:.2f}% of High Quality Rice".format(percentage_long_grain))
# # You can add more lines here for other categories if needed
# print("{:.2f}% of Low Quality Rice".format(percentage_broken_grain))


# Display the analysis based on major axis length
print("{:.2f}% of Very High Quality Rice".format(percentage_extra_long_grain))
print("{:.2f}% of High Quality Rice".format(percentage_long_grain))
print("{:.2f}% of  Average Quality Rice".format(percentage_medium_grain))
print("{:.2f}% of Low Quality Rice".format(percentage_short_grain))
print("{:.2f}% of Very Low Quality Rice".format(percentage_broken_grain))



# based on ratio of the seed

#ratio list
Aspect_ratio_of_seeds = []

# print("*******************Based On Aspect Ratio********************")
# Print the counts of each rice variety
# print("No of rice grains = ", len(contours))
# for variety, count in shape_varieties.items():
#     print(f"Count of {variety}: {count}")

total_seeds = len(contours)
percentage_slender_grain = (count_slender / total_seeds) * 100
percentage_medium_grain = (count_medium / total_seeds) * 100
percentage_bold_size_grain = (count_bold / total_seeds) * 100
percentage_round_grain = (count_round / total_seeds) * 100
print()

# print("**********Percentage of each category based on Aspect ratio *********")
# Print the calculated percentages
# print("Percentage of Slender Grain:", percentage_slender_grain)
# print("Percentage of Medium Grain:", percentage_medium_grain)
# print("Percentage of Bold Size Grain:", percentage_bold_size_grain)
# print("Percentage of Round Grain:", percentage_round_grain)
# print()

# Print  the analysis
print("**********Analysis based on the aspect ratio of the rice granules**********")
print()
# Display the analysis based on major axis length
# Display the analysis based on major axis length
print("{:.2f}% of Very High Quality Rice".format(percentage_slender_grain))
print("{:.2f}% of High Quality Rice".format(percentage_medium_grain))
print("{:.2f}% of Average Quality Rice".format(percentage_bold_size_grain))
print("{:.2f}% of Low Quality Rice".format(percentage_round_grain))



# broken seed
for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    # print(f"Area of Rice Seed {idx + 1}: {contour_area}")

areas = [cv2.contourArea(contour) for contour in contours]
# Calculate the mean area
mean_area = np.mean(areas)

# Print the mean area
# print("Mean Area of Rice Seeds:", mean_area)


# Calculate 75% of the mean area
three_fourths_mean_area = 0.75 * mean_area

# Print 75% of the mean area
# print("75% of Mean Area of Rice Seeds:", three_fourths_mean_area)

# Define classification thresholds
broken_threshold = 0.75 * mean_area
# print(broken_threshold)
fragment_threshold = 0.25 * mean_area

broken_count = 0
healthy_count = 0


for contour, area in zip(contours, areas):
    x, y, w, h = cv2.boundingRect(contour)
    if area <= broken_threshold or area <= fragment_threshold:
        # cv2.drawContours(dilation_color, [contour], -1, (255, 0, 0), 2)
        broken_count += 1
        cv2.rectangle(dilation_color_2, (x, y), (x+w, y+h), (0,255,0), 2)
    else:
        healthy_count += 1
# cv2.imwrite('output3.jpg', dilation_color)
print()
print("**********Analysis based on broken seed**********")
print("Broken count: ", broken_count)
print("Healthy count: ", healthy_count)

broken_percentage = (broken_count/healthy_count) * 100
print("Broken percentage: ", "{:.2f}".format(broken_percentage), "%")

def broken_quality(broken_percentage):
    rice_quality = ""
    if (broken_percentage <= 4.90):
        rice_quality =  "Very Good Quality"
    elif(4.90 <broken_percentage <= 19.75 ):
        rice_quality =  "Good Quality"
    elif(19.75 <broken_percentage <= 34.50):
        rice_quality =  "Low Quality"
    else:
        rice_quality =  "Very Bad Quality"
    return rice_quality

print("Quality of Rice : ", broken_quality(broken_percentage))
  

print()
# chalky detection
# # Load the image in grayscale
# image = cv2.imread('Images/img_01_norm.jpg', cv2.IMREAD_GRAYSCALE)

# # Estimate the background using morphological opening
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (190, 190))
# background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# # Subtract the background from the original image
# subtracted_image = cv2.subtract(image, background)

# # Increase contrast
# increased_contrast = cv2.convertScaleAbs(subtracted_image, alpha=1.5, beta=0)

# # Convert to binary image using adaptive thresholding
# _, binary_image = cv2.threshold(increased_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Apply K-means clustering
# num_clusters = 2
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# _, labels, centers = cv2.kmeans(np.float32(binary_image.reshape(-1, 1)), num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# # Separate chalky and non-chalky areas
# chalky_area = np.sum(labels == 0) if centers[0] < centers[1] else np.sum(labels == 1)
# total_rice_area = binary_image.shape[0] * binary_image.shape[1]


# # Calculate chalkiness percentage
# chalkiness_percentage = (chalky_area / total_rice_area) * 100

# # Determine chalkiness level
# chalkiness_level = "Affected" if chalkiness_percentage > 30 else "Good"
print("**********Analysis based on the Chalky level**********")
# print(f"Chalkiness Percentage: {chalkiness_percentage:.2f}%")
# print(f"Chalkiness Level: {chalkiness_level}")


# # Apply Canny edge detector to preprocess the image
# edges = cv2.Canny(img, 100, 200)

# # Create a structuring element for the extended maxima transform
# kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size

# # Apply morphological reconstruction
# reconstruction = cv2.dilate(edges, kernel)

# # Apply extended maxima transform
# extended_maxima = reconstruction - edges

# # Calculate the total area of chalky regions
# chalky_area = np.sum(extended_maxima == 255)

# # print("Chalky Area:", chalky_area)

# # Find contours in the extended maxima image
# contours, _ = cv2.findContours(extended_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Calculate the total area of rice seeds
# total_rice_area = 0
# for idx, contour in enumerate(contours):
#     contour_area = cv2.contourArea(contour)
#     # print(f"Area of Rice Seed {idx + 1}: {contour_area}")
#     total_rice_area += contour_area

# # print("Total area of Rice Seeds:", total_rice_area)

# chalky_percentage = (chalky_area / total_rice_area) *100
# # print("Chalky percentage: ", chalky_percentage)

# if(chalky_percentage > 30):
#     quality = "Bad Quality"
# else:
#     quality  = "Good Quality"

# print(f"Percentage of chalky area: {chalky_percentage:.2f}%")
# print(f"Quality: {quality}")





# seconddddddddddddddddddddddddddddddddddddddd methoddddddddddddddddddd
ret, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# con1 = cv2.drawContours(image, contours, -1, (0,255,0), 2)
contoursr, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# con2 = cv2.drawContours(image_copy, contoursr, -1, (255, 0, 0), 2)

# Calculate the total area of rice seeds
total_rice_area = 0

# Calculate Chalky Area
# chalky_area = np.sum(thresh == 255)
total_rice_area = 0
total_chalky = 0
for idx, contour in enumerate(contoursr):
    contour_area = cv2.contourArea(contour)
    # print(f"Area of Rice Seed {idx + 1}: {contour_area}")
    total_rice_area += contour_area

for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    # print(f"Area of  {idx + 1}: {contour_area}")
    total_chalky += contour_area

# print("Total area of Rice Seeds:", total_rice_area)
# print("Total chalky area: ", total_chalky)

chalky_percentage = (total_chalky/ total_rice_area) *100
# print("Chalky percentage: ", chalky_percentage)

if(chalky_percentage > 30):
    quality = "Bad Quality"
else:
    quality  = "Good Quality"

print(f"Percentage of chalky area: {chalky_percentage:.2f}%")
print(f"Quality: {quality}")

















# Draw contours on the original image
rice_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(rice_with_contours, contours, -1, (0, 0, 255), 2)



# impurity detection
# Convert the image to HSV color space
hsv_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Define the range of colors for potential impurities in HSV
lower_color = np.array([0, 50, 50])  # Lower HSV values for color range (modify as needed)
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
rice_with_contours_imp = img_rgb.copy()
cv2.drawContours(rice_with_contours_imp, contours, -1, (0, 0, 255), 2)

# Display the original image with contours
# cv2.imshow('Impurity Detection', rice_with_contours)

# Print the impurity percentage
print()
print("**********Analysis based on the impurity content**********")
print(f"Percentage of impurity area: {impurity_percentage:.2f}%")







plt.subplot(1,1,1)
plt.title("Original Image")
plt.axis("off")
plt.imshow(img_rgb)

# plt.subplot(2,3,2)
# plt.title("Count of rice grains")
# plt.axis("off")
# plt.imshow(contour_draw_count)

# plt.subplot(2,3,3)
# plt.title("Chalky Detection")
# plt.axis("off")
# plt.imshow(rice_with_contours)

# plt.subplot(2,3,4)
# plt.title("Impurity Detection")
# plt.axis("off")
# plt.imshow(rice_with_contours_imp)

# plt.subplot(2,3,5)
# plt.title("Broken Seed detection")
# plt.axis("off")
# plt.imshow(dilation_color_2)


# Define the percentages for each category
categories = [
    "High Quality Rice (Major Axis Length)",
    "Low Quality Rice (Major Axis Length)",
    "Broken Rice",
    "Chalky Rice",
    "Impurity Content",
]

percentages = [
    (percentage_extra_long_grain + percentage_long_grain),
    percentage_medium_grain,
    broken_count / total_seeds * 100,
    chalky_percentage,
    impurity_percentage,
]

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(percentages, labels=categories, autopct='%1.2f%%', startangle=140)
plt.title("Rice Quality Analysis Summary", fontsize = 16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the pie chart
plt.show()






