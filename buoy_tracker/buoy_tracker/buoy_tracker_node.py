from ament_index_python.packages import get_package_share_directory
from .buoy_detector import BuoyDetector
from .tracker import Trackers

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from tracker_msgs.msg import TrackerArray, Tracker
from sensor_msgs.msg import Image

class BuoyTracker(Node):
    def __init__(self) -> None:
        super().__init__('BuoyTrackerNode')
        package_name = 'buoy_tracker'
        package_path = get_package_share_directory(package_name)
        video_file_path = package_path + '/resource/data/detectbuoy.avi'
        weights_file_path = package_path + '/resource'

        self.declare_parameter('video_file_path', video_file_path)
        self.declare_parameter('weights_file_path', weights_file_path)
        self.declare_parameter('detect_freq', 2)
        self.declare_parameter('max_decay_count', 8)

        self.tracker_pub = self.create_publisher(TrackerArray, 'tracked_buoy', 10)
        self.tracker_image_publisher_ = self.create_publisher(Image, 'tracked_buoy_image', 10)

        video_file_path = self.get_parameter('video_file_path').get_parameter_value().string_value
        weights_file_path = self.get_parameter('weights_file_path').get_parameter_value().string_value
        
        self.detect_freq = self.get_parameter('detect_freq').get_parameter_value().integer_value
        self.max_decay_count = self.get_parameter('max_decay_count').get_parameter_value().integer_value

        if self.detect_freq >= self.max_decay_count:
            raise ValueError('detect_freq should be smaller than max_decay_count for tracker to function')

        self.trackers = Trackers(self.max_decay_count)
        self.buoy_detector = BuoyDetector(weights_file_path, weights_file_path)

        self.input_video = cv2.VideoCapture(video_file_path)
        self.frame_count = 0
        timer_period = 1/20  # seconds
        self.get_logger().info(f'Video file path: {video_file_path}', once=True)
        self.get_logger().info(f'Weights file path: {weights_file_path}', once=True)
        self.get_logger().info(f'Detect freq: {self.detect_freq}', once=True)
        self.get_logger().info(f'Max decay count: {self.max_decay_count}', once=True)

        self.timer = self.create_timer(timer_period, self.update)

    def update(self):
        if not self.input_video.isOpened():
            self.get_logger().info('Video not opened')
            self.input_video.release()
            self.destroy_node()
            exit(0)
        ret, frame = self.input_video.read()
        if not ret:
            self.get_logger().info('Read returned false')
            self.input_video.release()
            self.destroy_node()
            exit(0)
        
        if self.frame_count % self.detect_freq == 0:
            # print(f'Frame count: {self.frame_count}')
            detected_buoys = self.buoy_detector.detect(frame)
            frame = self.buoy_detector.draw_buoy(frame, detected_buoys)
        else:
            detected_buoys = {}
        
        if self.trackers.id == 0:
            for color in detected_buoys:
                # if color == 'orange':
                x, y, radius = detected_buoys[color][0]
                self.trackers.add_tracker(x, y)
            return
        
        z_list = []
        for color in detected_buoys:
            # if color == 'orange':
            z_list.extend([item[:2] for item in detected_buoys[color]])
        
        curr_states = self.trackers.update_and_predict(z_list)

        frame = self.trackers.draw_trackers(frame)
        
        text = "Green circle is prediction"
        position = (10, 50)
        frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        text = "Blue circle is update"
        position = (10, 100)
        frame = cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        bridge = CvBridge()
        ros_image = bridge.cv2_to_imgmsg(frame, 'bgr8')

        msg = TrackerArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        ros_image.header = msg.header
        for curr in curr_states:
            tracker = Tracker()
            tracker.id = curr
            tracker.x = curr_states[curr][0]
            tracker.y = curr_states[curr][1]
            tracker.radius = 20.0
            msg.trackers.append(tracker)

        self.tracker_pub.publish(msg)
        self.tracker_image_publisher_.publish(ros_image)
        self.frame_count += 1
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            self.input_video.release()
            self.destroy_node()
            exit(0)


def main(args=None):
    rclpy.init(args=args)
    buoyTrackerNode = BuoyTracker()
    rclpy.spin(buoyTrackerNode)
    buoyTrackerNode.get_logger().info('Node shutting down')
    rclpy.shutdown()


if __name__ == '__main__':
    main()





