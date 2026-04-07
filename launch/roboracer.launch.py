from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='roboracer',
            executable='perception_node',
            name='perception_node',
            output='screen',
        ),

        Node(
            package='roboracer',
            executable='decision_node',
            name='decision_node',
            output='screen',
        ),

        Node(
            package='roboracer',
            executable='control_node',
            name='control_node',
            output='screen',
        ),

    ])
