from buoy_detector import BuoyDetector
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
# from filterpy.kalman import ExtendedKalmanFilter

class EKFTracker:
    def __init__(self, x, y, id, max_decay_count) -> None:
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
        self.Q = np.array([[0.05, 0, 0, 0, 0, 0],
                           [0, 0.03, 0, 0, 0, 0],
                           [0, 0, 0.07, 0, 0, 0],
                           [0, 0, 0, 0.05, 0, 0],
                           [0, 0, 0, 0, 0.03, 0],
                           [0, 0, 0, 0, 0, 0.07]])
        self.R = np.array([[0.05, 0],
                            [0, 0.05]])
        # self.previous_x = None
        self.id = id
        self.measurements_count = 0
        self.predict_count = 0
        self.max_decay_count = max_decay_count
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.predict_count += 1
        validity = True
        if abs(self.measurements_count - self.predict_count) > self.max_decay_count:
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

    def add_tracker(self, x, y, max_decay_count):
        self.trackers[self.id] = EKFTracker(x, y, self.id, max_decay_count)
        self.predicted_states[self.id] = np.array([x,y])
        self.id += 1

    def associate_and_update(self, z_list):
        predicted_positions_list = [self.predicted_states[key] for key in sorted(self.predicted_states.keys())]
        cost_matrix = distance_matrix(predicted_positions_list, z_list)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row, col in zip(row_indices, col_indices):
            tracker_id = sorted(self.predicted_states.keys())[row]
            z = z_list[col]
            state = self.trackers[tracker_id].update(z)
            self.updated_states[tracker_id] = np.array([state[0], state[3]])

    def predict(self):
        delete_ids = []
        for tracker_id in self.trackers:
            state, validity = self.trackers[tracker_id].predict()
            if not validity:
                delete_ids.append(tracker_id)
            else:
                self.predicted_states[tracker_id] = np.array([state[0], state[3]])
        for tracker_id in delete_ids:
            del self.trackers[tracker_id]
            del self.predicted_states[tracker_id]
            del self.updated_states[tracker_id]

    def draw_trackers(self, frame):
        for tracker_id in self.trackers:
            predicted_state = self.predicted_states[tracker_id].astype(int)
            radius = 20
            cv2.circle(frame, predicted_state, radius, (0, 255, 0), 2) # Prediction is green
            updated_state = self.updated_states[tracker_id].astype(int)
            cv2.circle(frame, updated_state, radius, (255, 0, 0), 2) # Update is blue
            # Add tracker id using putText
            cv2.putText(frame, str(tracker_id), (int(updated_state[0] + 10), int(updated_state[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if np.array_equal(predicted_state, updated_state):
                continue
            vec = predicted_state - updated_state
            norm_vec = (vec / np.linalg.norm(vec)).astype(int) * 20
            norm_vec = updated_state + norm_vec
            print(f'Norm direction vector is {norm_vec}')
            cv2.arrowedLine(frame, updated_state, norm_vec, (0, 0, 255), 2, tipLength=0.5)

        return frame

    def get_tracker(self, tracker_id):
        return self.trackers[tracker_id]

    def update_and_predict(self, z_list):
        curr_states = None
        if z_list != []:
            self.associate_and_update(z_list)
            curr_states = self.updated_states
        else:
            curr_states = self.predicted_states
        self.predict()
        return curr_states


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
            trackers.add_tracker(x, y, max_decay_count)
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
