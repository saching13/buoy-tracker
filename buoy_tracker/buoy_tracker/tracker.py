import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix

class EKFTracker:
    def __init__(self, x, y, id, max_decay_count) -> None:
        dt = 1/40
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
    
        self.Q = np.array([[0.05, 0, 0, 0, 0, 0],
                           [0, 0.03, 0, 0, 0, 0],
                           [0, 0, 0.07, 0, 0, 0],
                           [0, 0, 0, 0.05, 0, 0],
                           [0, 0, 0, 0, 0.03, 0],
                           [0, 0, 0, 0, 0, 0.07]])
        self.R = np.array([[0.02, 0],
                            [0, 0.02]])
        # self.previous_x = None
        self.id = id
        self.age = 0
        self.max_decay_count = max_decay_count
    
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        validity = True
        if self.age > self.max_decay_count:
            validity = False
        return self.x, validity

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z - self.H @ self.x
        # print('y is ', y)
        # print('K is ', K)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P
        self.age = 0
        return self.x


class Trackers:
    def __init__(self, max_decay_count) -> None:
        self.trackers = {}
        self.id = 0
        self.predicted_states = {}
        self.updated_states = {}
        self.max_decay_count = max_decay_count

    def add_tracker(self, x, y):
        self.trackers[self.id] = EKFTracker(x, y, self.id, self.max_decay_count)
        self.predicted_states[self.id] = np.array([x,y])
        self.updated_states[self.id] = np.array([x,y])
        self.id += 1

    def associate_and_update(self, z_list):
        predicted_positions_list = [self.predicted_states[key] for key in sorted(self.predicted_states.keys())]
        print(f'Size if Z list is {len(z_list)}')
        print(f'Size of predicted positions list is {len(predicted_positions_list)}')
        cost_matrix = distance_matrix(predicted_positions_list, z_list)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        print(f'cost matrix is \n {cost_matrix}')
        print(f'row indices are {row_indices}')
        print(f'col indices are {col_indices}')
        for row, col in zip(row_indices, col_indices):
            tracker_id = sorted(self.predicted_states.keys())[row]
            z = z_list[col]
            state = self.trackers[tracker_id].update(z)
            self.updated_states[tracker_id] = np.array([state[0], state[3]])
        if len(col_indices) < len(z_list):
            for col in range(len(z_list)):
                if col not in col_indices:
                    x, y = z_list[col]
                    self.add_tracker(x, y)

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

