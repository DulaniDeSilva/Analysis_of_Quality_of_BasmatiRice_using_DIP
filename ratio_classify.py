import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'

#loading the image in grayscale
# img_bgr = cv2.imread("imp_9.jpg", 1)
img_bgr = cv2.imread("Images/1.jpg", 1)
# img_bgr = cv2.imread("Images/img_01_norm.jpg", 1)
# img_bgr = cv2.imread("Images/img_01_pure.jpg", 1)

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
print("No of rice grains = ", len(contours))

contour_draw = cv2.drawContours(dilation_color, contours, -1, (0,255,0), 2)

#initialize list to store axis lengths
major_axis_length = []
minor_axis_length = []

#ratio list
Aspect_ratio_of_seeds = []

count_slender = 0
count_medium = 0
count_bold = 0
count_round = 0


def classification_ratio(ratio):
    global count_slender, count_medium, count_bold, count_round 
    ratio_class = ""
    if ratio >= 3:
        ratio_class = "Slender Grain Shape"
        count_slender += 1
        print("Slender Shape count: ", count_slender)
    elif(ratio < 3 and ratio >= 2.1):
        ratio_class = "Medium Grain Shape"
        count_medium += 1
        print("Medium Shape count: ", count_medium)
    elif(ratio < 2.1 and ratio >= 1.1):
        ratio_class = "Bold Size Grain"
        count_bold += 1
        print("Bold Shape count: ", count_bold)
    else:
        ratio_class = "Round Grain"
        count_round += 1
        print("Slender Shape count: ", count_round)
    return ratio_class


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

    print("Seed Number: ", seed_number, " |", "Aspect ratio",aspect_ratio, "|", "Classification According to ratio: ", classification_ratio(aspect_ratio))
   
    major_axis_length.append(major_axis)
    minor_axis_length.append(minor_axis)


print()

shape_varieties = {
    "Slender Grain": count_slender,
    "Medium Grain": count_medium,
    "Bold Size Grain": count_bold,
    "Round Grain": count_round
}
print()
print("*******************Count********************")
# Print the counts of each rice variety
print("No of rice grains = ", len(contours))
for variety, count in shape_varieties.items():
    print(f"Count of {variety}: {count}")

total_seeds = len(contours)
percentage_slender_grain = (count_slender / total_seeds) * 100
percentage_medium_grain = (count_medium / total_seeds) * 100
percentage_bold_size_grain = (count_bold / total_seeds) * 100
percentage_round_grain = (count_round / total_seeds) * 100
print()

print("**********Percentage of each category *********")
# Print the calculated percentages
print("Percentage of Slender Grain:", percentage_slender_grain)
print("Percentage of Medium Grain:", percentage_medium_grain)
print("Percentage of Bold Size Grain:", percentage_bold_size_grain)
print("Percentage of Round Grain:", percentage_round_grain)
print()

# Print  the analysis
print("**********Analysis based on the aspect ratio of the rice granules**********")

# Display the analysis based on major axis length
print("{:.2f}% of Very High Quality Rice".format(percentage_slender_grain))
print("{:.2f}% of High Quality Rice".format(percentage_medium_grain))
# You can add more lines here for other categories if needed
print("{:.2f}% of Average Quality Rice".format(percentage_bold_size_grain))
print("{:.2f}% of Low Quality Rice".format(percentage_round_grain))



plt.subplot(1,2,1)
plt.axis("off")
plt.title("Original Image", color = COLOR, fontsize = FONT_SIZE)
plt.imshow(img_rgb)

# plt.subplot(1,3,2)
# plt.axis("off")
# plt.title("Quality based on Aspect Ratio",color = COLOR, fontsize = FONT_SIZE)
# plt.imshow(dilation_color)

plt.subplot(1,2,2)
# Counts of each classification ratio
counts = [percentage_slender_grain, percentage_medium_grain, percentage_bold_size_grain, percentage_round_grain]
counts = [count_slender, count_medium, count_bold, count_round]
# Classification ratio labels
labels = ["Slender Grain", "Medium Grain", "Bold Size Grain", "Round Grain"]
# Colors for each category
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
# Explode a slice if needed (e.g., 'explode' Slender Grain)
explode = (0.1, 0, 0, 0)  # Only "explode" the 1st slice (i.e., Slender Grain)
plt.pie(counts, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Distribution of Classification Ratios", color = "maroon", fontsize = 8)

plt.show()








