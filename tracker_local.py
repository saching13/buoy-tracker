from buoy_detector import BuoyDetector
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
# from filterpy.kalman import ExtendedKalmanFilter

class EKFTracker:
    def __init__(self, x, y, id, valid_threshold) -> None:
        dt = 1/30
        self.x = np.array([x, 0, 0, y, 0, 0])
        self.F = np.array([[1, dt, 0.5*dt**2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, 0.5*dt**2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])
        self.dt = dt
        print(f'Shape of x is {self.x.shape}')
        self.P = np.eye(self.x.shape[0])
        self.Q = np.array([[0.07, 0, 0, 0, 0, 0],
                           [0, 0.03, 0, 0, 0, 0],
                           [0, 0, 0.07, 0, 0, 0],
                           [0, 0, 0, 0.07, 0, 0],
                           [0, 0, 0, 0, 0.03, 0],
                           [0, 0, 0, 0, 0, 0.07]])
        self.R = np.array([[0.05, 0],
                            [0, 0.05]])
        # self.previous_x = None
        self.id = id
        self.measurements_count = 0
        self.predict_count = 0
        self.valid_threshold = valid_threshold
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.predict_count += 1
        validity = True
        if abs(self.measurements_count - self.predict_count) > self.valid_threshold:
            validity = False
        return self.x, validity

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        self.measurements_count += 1
        return self.x


class Trackers:
    def __init__(self) -> None:
        self.trackers = {}
        self.id = 0
        self.predicted_states = {}
        self.updated_states = {}

    def add_tracker(self, x, y, valid_threshold):
        self.trackers[self.id] = EKFTracker(x, y, self.id, valid_threshold)
        self.id += 1

    def associate_and_update(self, z_list):
        predicted_positions_list = [self.predicted_states[key] for key in sorted(self.predictions.keys())]
        cost_matrix = distance_matrix(predicted_positions_list, z_list)
        print(f'Cost Matrix is \n {cost_matrix}')
        row_indices, col_indices = linear_sum_assignment(cost_matrix)


        for row, col in zip(row_indices, col_indices):
            print(f'Selected cost value are {cost_matrix[row, col]}')
            tracker_id = sorted(self.predicted_states.keys())[row]
            z = z_list[col]
            state = self.trackers[tracker_id].update(z)
            self.updated_states[tracker_id] = np.array([state[0], state[3]])

    def predict(self):
        for tracker_id in self.trackers:
            state, validity = self.trackers[tracker_id].predict()
            if not validity:
                del self.trackers[tracker_id]
                del self.predicted_states[tracker_id]
            else:
                self.predicted_states[tracker_id] = np.array([state[0], state[3]])

    def draw_trackers(self, frame):
        for tracker_id in self.trackers:
            predicted_state = self.predicted_states[tracker_id]
            cv2.circle(frame, predicted_state, 10, (0, 255, 0), 2) # Prediction is green
            updated_state = self.updated_states[tracker_id]
            cv2.circle(frame, updated_state, 10, (255, 0, 0), 2) # Update is blue
            # Add tracker id using putText
            cv2.putText(frame, str(tracker_id), (int(updated_state[0] + 10), int(updated_state[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def get_tracker(self, tracker_id):
        return self.trackers[tracker_id]


ekf = None

filename = 'data/detectbuoy.avi'
input_video = cv2.VideoCapture(filename)
# input_video.set(1,start_frame)
buoy_detector = BuoyDetector()
isSet = False
while input_video.isOpened():
    ret, frame = input_video.read()
    if not ret:
        break
    
    detected_buoys = buoy_detector.detect(frame)
    if ekf is None and 'orange' in detected_buoys:
        x, y, radius = detected_buoys['orange'][0]
        # ekf.x = np.array([x, 0, 0, y, 0, 0])  # initial state
        ekf = EKFTracker(x, y, 'orange', 10)
    else:
        x, y, radius = detected_buoys['orange'][0]
        ekf.update(np.array([x, y]))
        updated_pose = ekf.x
        ekf.predict()

        predicted_pose = ekf.x
        print('Printing Updated and predicted pose')
        cv2.circle(frame, (int(predicted_pose[0]), int(predicted_pose[3])), int(radius), (0, 255, 0), 2)
        cv2.circle(frame, (int(updated_pose[0]), int(updated_pose[3])), int(radius), (255, 0, 0), 2)

        print(updated_pose, predicted_pose)
    frame = buoy_detector.draw_buoy(frame, detected_buoys)

    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    
    if k == ord('q'):
        input_video.release()
