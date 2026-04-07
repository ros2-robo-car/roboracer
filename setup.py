from setuptools import find_packages, setup

package_name = 'roboracer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/roboracer.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Roboracer Physical AI 자율주행 패키지',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = roboracer.perception_node:main',
            'decision_node   = roboracer.decision_node:main',
            'control_node    = roboracer.control_node:main',
        ],
    },
)
