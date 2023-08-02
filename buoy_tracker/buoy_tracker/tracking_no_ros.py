from buoy_detector import BuoyDetector
import cv2
import numpy as np
from tracker import Trackers

trackers = Trackers()
filename = 'data/detectbuoy.avi'
input_video = cv2.VideoCapture(filename)

buoy_detector = BuoyDetector()
isSet = False
detect_freq = 10
count = 0

while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break
    if count % detect_freq == 0:
        detected_buoys = buoy_detector.detect(frame)
    else:
        detected_buoys = {}
    

    if trackers.id == 0:
        for color in detected_buoys:
            x, y, radius = detected_buoys[color][0]
            max_decay_count = 5
            trackers.add_tracker(x, y)
        continue

    # Abandoning the information of the associated color to demonstrate the association
    z_list = []
    for color in detected_buoys:
        # print(item[:2]) for item in detected_buoys[color]
        z_list.extend([item[:2] for item in detected_buoys[color]])
    
    curr_states = trackers.update_and_predict(z_list)
    # trackers.associate_and_update(z_list)
    # trackers.predict()

    frame = trackers.draw_trackers(frame)
    frame = buoy_detector.draw_buoy(frame, detected_buoys)

    text = "Green circle is prediction"
    position = (10, 50)
    frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    text = "Blue circle is update"
    position = (10, 100)
    frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    
    if k == ord('q'):
        input_video.release()
