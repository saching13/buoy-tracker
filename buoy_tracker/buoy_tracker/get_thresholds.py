import cv2
import numpy as np

# Load image
image = cv2.imread('/home/sachin/luxonis/tracking/buoy_tracker/data/TrainingSet/Frames/001.jpg')

# Function to apply mask
def apply_mask(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)

# Function to handle slider movement
def slider_handler(x):
    lower_bound = np.array([cv2.getTrackbarPos('LowB', 'image'),
                            cv2.getTrackbarPos('LowG', 'image'),
                            cv2.getTrackbarPos('LowR', 'image')])
    upper_bound = np.array([cv2.getTrackbarPos('HighB', 'image'),
                            cv2.getTrackbarPos('HighG', 'image'),
                            cv2.getTrackbarPos('HighR', 'image')])
    masked_image = apply_mask(image, lower_bound, upper_bound)
    cv2.imshow('image', masked_image)

# Create window and sliders
cv2.namedWindow('image')
cv2.createTrackbar('LowB', 'image', 0, 255, slider_handler)
cv2.createTrackbar('HighB', 'image', 255, 255, slider_handler)
cv2.createTrackbar('LowG', 'image', 0, 255, slider_handler)
cv2.createTrackbar('HighG', 'image', 255, 255, slider_handler)
cv2.createTrackbar('LowR', 'image', 0, 255, slider_handler)
cv2.createTrackbar('HighR', 'image', 255, 255, slider_handler)

# Show original image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
