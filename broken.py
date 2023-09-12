import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT_SIZE = 8
COLOR = 'maroon'

# Load the image
img_bgr = cv2.imread("Images/3.jpg", 1)
# img_bgr = cv2.imread("imp_4.jpg", 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#convertion into binary
ret, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

#averaging filter
kernel = np.ones((3,3), np.float32)/9
dst = cv2.filter2D(thresh, -1, kernel)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

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

# contour_draw = cv2.drawContours(dilation_color, contours, -1, (0,255,0), 2)

for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)
    print(f"Area of Rice Seed {idx + 1}: {contour_area}")

areas = [cv2.contourArea(contour) for contour in contours]
# Calculate the mean area
mean_area = np.mean(areas)
print()
# Print the mean area
print("Mean Area of Rice Seeds:", mean_area)


# Calculate 75% of the mean area
three_fourths_mean_area = 0.75 * mean_area
print()
# Print 75% of the mean area
print("75% of Mean Area of Rice Seeds:", three_fourths_mean_area)
print()
# Define classification thresholds
broken_threshold = 0.75 * mean_area
print(broken_threshold)
fragment_threshold = 0.25 * mean_area

broken_count = 0
healthy_count = 0


# Classify rice kernels and draw contours
# for contour, area in zip(contours, areas):
#     if area <= broken_threshold:
#         cv2.drawContours(dilation_color, [contour], -1, (255, 0, 0), 2)  # Blue contour for fragment
#         broken_count+=1
#     else:
#         cv2.drawContours(dilation_color, [contour], -1, (0, 255, 0), 2)  # Green contour for quality
#         healthy_count+=1
# print()

for contour, area in zip(contours, areas):
    x, y, w, h = cv2.boundingRect(contour)
    if area <= broken_threshold or area <= fragment_threshold:
        cv2.drawContours(dilation_color, [contour], -1, (255, 0, 0), 2)
        broken_count += 1
        cv2.rectangle(dilation_color, (x, y), (x+w, y+h), (0,255,0), 2)
    else:
        healthy_count += 1
cv2.imwrite('output3.jpg', dilation_color)


print("Broken count: ", broken_count)
print("Healthy count: ", healthy_count)
print()
print()

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
    



# Display the original image with contours
plt.subplot(1,3,1)
plt.axis("off")
plt.title("Rice Sample ", color ="maroon", fontsize = 12)
plt.imshow(img_rgb)

plt.subplot(1,3,2)
plt.axis("off")
plt.title("Broken Seed Detection ", color ="maroon", fontsize = 12)
plt.imshow(dilation_color)

plt.subplot(1,3,3)
labels = ['Broken', 'Healthy']
sizes = [broken_count, healthy_count]
colors = ['red', 'green']
explode = (0.1, 0)  
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Distribution of Broken and Healthy Counts', color = "maroon", fontsize = 8)


plt.show()


