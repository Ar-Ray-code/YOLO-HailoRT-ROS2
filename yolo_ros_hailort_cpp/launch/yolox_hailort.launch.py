# Copyright 2023 Ar-Ray-code
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video0',
            description='input video source'
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value='./src/hailort_yolo_common/weights/yolox_tiny.hef',
            description='HEF model path.'
        ),
        DeclareLaunchArgument(
            'class_labels_path',
            default_value='',
            description='if use custom model, set class name labels. '
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='0.30',
            description='yolo confidence threshold.'
        ),
        DeclareLaunchArgument(
            'nms',
            default_value='0.45',
            description='yolo nms threshold'
        ),
        DeclareLaunchArgument(
            'imshow_isshow',
            default_value='true',
            description=''
        ),
        DeclareLaunchArgument(
            'src_image_topic_name',
            default_value='/image_raw',
            description='topic name for source image'
        ),
        DeclareLaunchArgument(
            'publish_image_topic_name',
            default_value='/yolo/image_raw',
            description='topic name for publishing image with bounding box drawn'
        ),
        DeclareLaunchArgument(
            'publish_boundingbox_topic_name',
            default_value='/yolo/bounding_boxes',
            description='topic name for publishing bounding box message.'
        ),
        DeclareLaunchArgument(
            'publish_resized_image',
            default_value='false',
            description='use BoundingBoxArray message type.'
        ),
    ]
    container = ComposableNodeContainer(
        name='yolo_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='usb_cam',
                plugin='usb_cam::UsbCamNode',
                name='usb_cam_node',
                parameters=[{
                    'video_device': LaunchConfiguration('video_device'),
                    'brightness': 100
                }]),
            ComposableNode(
                package='yolo_ros_hailort_cpp',
                plugin='yolo_ros_hailort_cpp::YoloNode',
                name='yolo_ros_hailort_cpp',
                parameters=[{
                    'model_path': LaunchConfiguration('model_path'),
                    'class_labels_path': LaunchConfiguration('class_labels_path'),
                    'conf': LaunchConfiguration('conf'),
                    'nms': LaunchConfiguration('nms'),
                    'imshow_isshow': LaunchConfiguration('imshow_isshow'),
                    'src_image_topic_name': LaunchConfiguration('src_image_topic_name'),
                    'publish_image_topic_name': LaunchConfiguration('publish_image_topic_name'),
                    'publish_boundingbox_topic_name': LaunchConfiguration('publish_boundingbox_topic_name'),
                    'publish_resized_image': LaunchConfiguration('publish_resized_image'),
                }],
                ),
        ],
        output='screen',
    )

    return launch.LaunchDescription(
        launch_args +
        [
            container
        ]
    )