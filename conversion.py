import cv2

# Load the rice image
img = cv2.imread("Images/con.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise (adjust kernel size as needed)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image to create a binary image
_, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it corresponds to a whole rice seed)
largest_contour = max(contours, key=cv2.contourArea)

# Calculate the length of the rice seed (assuming it's the major axis)
x, y, w, h = cv2.boundingRect(largest_contour)
length_pixels = max(w, h)

# Display the length in pixels
print(f"Length of one rice seed in pixels: {length_pixels}")

# Optionally, you can draw the contour on the image to visualize it
cv2.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
cv2.imshow("Rice Seed Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
