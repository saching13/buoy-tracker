from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    package_name = 'buoy_tracker'
    package_path = get_package_share_directory(package_name)
    video_file_path_str = package_path + '/resource/data/detectbuoy.avi'
    weights_file_path_str = package_path + '/resource'

    video_file_path = LaunchConfiguration('video_file_path',
                                          default=video_file_path_str)
    weights_file_path = LaunchConfiguration('weights_file_path',
                                            default=weights_file_path_str)
    detect_freq = LaunchConfiguration('detect_freq', default=2)
    max_decay_count = LaunchConfiguration('max_decay_count', default=8)

    # Declare the launch arguments
    declare_video_file_path_cmd = DeclareLaunchArgument(
        'video_file_path',
        default_value=video_file_path,
        description=
        'Provide the absolute path to the video file used for detection.')

    declare_weights_file_path_cmd = DeclareLaunchArgument(
        'weights_file_path',
        default_value=weights_file_path,
        description=
        'Provide the path to the folder which has weights for the detection algo (GMM).'
    )

    declare_detect_freq_cmd = DeclareLaunchArgument(
        'detect_freq',
        default_value=detect_freq,
        description=
        'detect freq is used to set at what intervel we run the detection instead of all the frames. .'
    )

    declare_max_decay_count_cmd = DeclareLaunchArgument(
        'max_decay_count',
        default_value=max_decay_count,
        description=
        'Max decay count is the number of frames a detection is not provided to the tracker before the track is considered unusable / discarded.'
    )

    # Create the node
    byoy_node = Node(package='buoy_tracker', executable='buoy_tracker_node',
                     output='screen',
                     parameters=[{'video_file_path': video_file_path},
                                 {'weights_file_path': weights_file_path},
                                 {'detect_freq': detect_freq},
                                 {'max_decay_count': max_decay_count}])

    ld = LaunchDescription()
    ld.add_action(declare_video_file_path_cmd)
    ld.add_action(declare_weights_file_path_cmd)
    ld.add_action(declare_detect_freq_cmd)
    ld.add_action(declare_max_decay_count_cmd)

    ld.add_action(byoy_node)
    return ld