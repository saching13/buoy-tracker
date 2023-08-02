import numpy as np
from sklearn.mixture import GaussianMixture
import os
import cv2

images = []
files_path = "/home/sachin/luxonis/tracking/buoy_tracker/data/TrainingSet/Frames/"
for filename in os.listdir(files_path):
    if 'db' in filename:
        continue
    print(filename)
    img = cv2.imread(files_path + '/' + filename)
    image = cv2.resize(img,(0, 0), fx= 0.5, fy=0.5)
    images.append(image)

print(f'Resized image size is {images[0].shape}')

# Reshape images
reshaped_images = [image.reshape(-1, 1) for image in images]

# Concatenate images
dataset = np.concatenate(reshaped_images, axis=0)

# Fit Gaussian Mixture model
gm = GaussianMixture(n_components=4)
gm.fit(dataset)

test_img = images[0].reshape(-1, 1)
predictions = gm.predict(test_img)
resize_prediction = predictions.reshape(images[0].shape)
max_val = np.max(resize_prediction)
resize_prediction = resize_prediction / max_val * 255 
resize_prediction = resize_prediction.astype(np.uint8)
cv2.imshow("res_img", resize_prediction)
cv2.waitKey(0)