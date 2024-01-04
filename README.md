# Averera
![vehicles](vehicle.jpg)

# Self-Driving Test Vehicle

The Autonomous Self-Driving Test Vehicle project combines computer vision, perception algorithms, and advanced control systems to achieve autonomous navigation. It integrates perception, localization, path planning, actuators, and real-time hardware communication to create a comprehensive self-driving solution.

## Project Overview

This project aims to build a self-driving vehicle capable of perceiving its environment, planning paths, and executing actions autonomously. It incorporates several key components:

- **Computer Vision and Perception:** The project uses advanced computer vision techniques for road segmentation, obstacle detection, and environment understanding.

- **Localization:** Precise localization is ensured by sensors and camera data to determine the vehicle's position accurately.

- **Path Planning:** Path planning algorithms generate optimal trajectories based on environment perception and destination, creating waypoints for navigation.

- **Control Systems:** PID controllers offer real-time control for accurate steering, speed regulation, and overall stability.

- **Arduino Communication:** ROS and rosserial enable seamless communication with Arduino microcontrollers, allowing physical interactions and environment changes.

## Features and Highlights

- **Camera Feed Integration:** Real-time camera feeds from Logitech USB cameras and Oak-D Lite cameras provide critical visual data for perception.

- **Joystick Control:** ROS-enabled joystick control allows manual interaction and controlled testing of vehicle functions.

- **Deep Learning Framework:** TensorFlow integration facilitates deep learning models for advanced perception tasks and object detection.

## Usage

1. Install necessary ROS packages, including USB camera drivers and joystick support.
2. Launch camera nodes to feed video data into the perception pipeline.
3. Execute the perception node (`vision.py`) for road segmentation and object detection.
4. Generate waypoints and path plans using control systems and algorithms.
5. Utilize PID controllers for precise steering and speed management.
6. Communicate with Arduino microcontrollers for physical actuation.

## Results

The project presents two critical demonstrations. The first showcases road segmentation and perception through a GIF, while the second displays autonomous vehicle navigation using perception, control, and planning.

![Segmentation Results](wda_AdobeExpress.gif)

![Autonomous Operation](aaa.gif)

![3D mapping by SLAM](map.png)



https://github.com/Atharv-16/Averera/assets/119600601/591098bb-d76f-4b30-8962-95f5171b66fb



https://github.com/Atharv-16/Averera/assets/119600601/cb1b2a9f-1cd6-413d-8ec0-e0f41c27451d






## Future Development

As autonomous technology advances, this project serves as a foundation for further exploration. Future enhancements may involve advanced machine learning, semantic mapping, obstacle avoidance, and integration with larger-scale autonomous systems.

The Autonomous Self-Driving Test Vehicle project demonstrates the limitless potential of modern robotics and automation.

## Running

1. `roslaunch usb_cam usb_cam-test.launch` publishes camera frame to "/usb_cam/image_raw" topic.
2. `vision.py` subscribes to "/usb_cam/image_raw" and publishes error to "error" topic.
3. `controller.py` subscribes to "error" and publishes to "steer" topic.
4. `steer.py` subscribes to "steer_arduino" topic.
5. `throttle.py` gives constant throttle and publishes to "arduino" topic.

`serial_node_steer.py` and `serial_node_throttle.py` used for communication with arduino using rosserial.

`steer_joy.py` is to be used for joystick control only.

Download model weights from [here](https://drive.google.com/file/d/1dZHzGoYTvhoo_Ys4Sx4edBHCpdC61raI/view?usp=sharing).
Download Manual control ROS package from [here](https://drive.google.com/file/d/1dvYleEhMm6OSeBeS5bw1i_uCxmsdZ_JS/view?usp=sharing).
Download Autonomous ROS package from [here](https://drive.google.com/file/d/1FMS4M6jBH44xCsQ8LMSfTpOO6fT5P897/view?usp=sharing).
Download Oakd-Lite ROS package from [here](https://drive.google.com/file/d/17iS2hzYX-NaiL9oLPft_460quamsX5fC/view?usp=sharing).
## Setting up

1. Run `sudo apt-get install ros-noetic-usb-cam` to install camera node for video feed
2. Run `sudo apt-get install ros-noetic-joy` on terminal to install joystick package.
3. Run `rosrun joy joy_node` to start the node.
4. Run `rostopic echo joy` to see the topic result.
5. Run `rosrun usb_cam usb_cam_node` for the camera feed for logitech usb cam feed.
6. Run `roslaunch depthai_examples mobile_publisher.launch` for oak-d lite cam feed.

7. Run `python3 -m pip install tensorflow` and `pip install tensorflow-addons` on terminal to install tensorflow for `vision.py`.
8. Run `sudo apt-get install ros-noetic-rosserial-arduino` to install rosserial for communication with arduino.

## Setting up Oak-D Lite:

1. Follow steps from  [here](https://github.com/luxonis/depthai-ros).
2. Follow every step till Docker(excluded), then go to 'Executing a file'.
3. There a 8 packages , do not use catkin_build_isolated but use catkin build.
4. After build, simply launch file given above at point 6 in Setting up.
5. Multiple topics will appear and desired image will be published on some topic.

## General Structure of Autonomy

![flowchart](https://github.com/Atharv-16/Averera/assets/119600601/10c70a10-1a8e-4d39-b091-d55117b2fcaa)



