import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'


#loading the image in grayscale
# img_bgr = cv2.imread("Images/img_01_pure.jpg", 1)
img_bgr = cv2.imread("Images/3.jpg", 1)
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

count_extra_long = 0
count_long = 0
count_medium = 0 
count_short = 0
count_broken = 0

def classification_length(axis_length):
    global count_extra_long, count_long, count_medium, count_short, count_broken
    length_class = ""
    if axis_length >= 85:
        length_class = "Extra Long Grain"
        count_extra_long += 1
        print("count_extra_long:", count_extra_long)
    elif(axis_length < 85 and axis_length >= 60):
        length_class = "Long Grain"
        count_long += 1
        print("count_long:", count_long)
    elif(axis_length < 60 and axis_length >= 50):
        length_class = "Medium  Grain"
        count_medium += 1
        print("count_medium:", count_medium)
    elif (axis_length < 50 and axis_length >= 30):
        length_class = "Short Grain"
        count_short += 1
        print("count_short:", count_short)
    else:
        length_class = "Broken Grain"
        count_broken += 1
        print("count_broken:", count_broken)

    cv2.putText(dilation_color, "Extra Long: " + str(count_extra_long), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(dilation_color, "Long: " + str(count_long), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(dilation_color, "Medium: " + str(count_medium), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(dilation_color, "Short: " + str(count_short), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(dilation_color, "Broken: " + str(count_broken), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return length_class


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

    print("Seed Number: ", seed_number, " | ", 
              "Major axis length: ", major_axis, "|", 
              "Minor axis length: ", minor_axis, "|", 
              "Accoridng to length: ", classification_length(major_axis))
        
    major_axis_length.append(major_axis)
    minor_axis_length.append(minor_axis)


# for idx, contour in enumerate(contours):
#     if len(contour) >=5:
#         seed_number = num_seeds - idx

#         ellipse = cv2.fitEllipse(contour)
#         center, axes, angle = ellipse

#         major_axis = max(axes) 
#         minor_axis = min(axes) 

       
#         print("Seed Number: ", seed_number, " | ", 
#               "Major axis length: ", major_axis, "|", 
#               "Minor axis length: ", minor_axis, "|", 
#               "Accoridng to length: ", classification_length(major_axis))
        
#         major_axis_length.append(major_axis)
#         minor_axis_length.append(minor_axis)

print()
rice_varieties = {
    "Extra Long Grain": count_extra_long,
    "Long Grain": count_long,
    "Medium Grain": count_medium,
    "Short Grain": count_short,
    "Broken Grain": count_broken
}
print("********************Counts ********************")
print()
# Print the counts of each rice variety
print("No of rice grains = ", len(contours))
for variety, count in rice_varieties.items():
    print(f"Count of {variety}: {count}")

total_seeds = len(contours)
percentage_extra_long_grain = (count_extra_long / total_seeds) * 100
percentage_long_grain = (count_long / total_seeds) * 100
percentage_medium_grain = (count_medium / total_seeds) * 100
percentage_short_grain = (count_short / total_seeds) * 100
percentage_broken_grain = (count_broken / total_seeds) * 100

print()
# Print  the analysis
print("**********Analysis based on the major axis length of the rice granules**********")

# Display the analysis based on major axis length
print("{:.2f}% of Very High Quality Rice".format(percentage_extra_long_grain))
print("{:.2f}% of High Quality Rice".format(percentage_long_grain))
print("{:.2f}% of  Average Quality Rice".format(percentage_medium_grain))
print("{:.2f}% of Low Quality Rice".format(percentage_short_grain))
# You can add more lines here for other categories if needed
print("{:.2f}% of Very Low Quality Rice".format(percentage_broken_grain))


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

plt.subplot(2,3,5)

quality_categories = ["Very High Quality", "High Quality", "Average Quality", "Low Quality", "Very Low Quality"]
quality_percentages = [percentage_extra_long_grain, percentage_long_grain, percentage_medium_grain, percentage_short_grain,percentage_broken_grain]
plt.pie(quality_percentages, labels=quality_categories, autopct='%.2f%%', colors=['green', 'blue', 'purple','yellow','red'])
plt.title("Quality Percentage of Rice Sample", color = "maroon")

plt.show()

