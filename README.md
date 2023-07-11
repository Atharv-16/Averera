# Averera

1. `roslaunch usb_cam usb_cam-test.launch` publishes camera frame to "/usb_cam/image_raw" topic.
2. `vision.py` subscribes to "/usb_cam/image_raw" and publishes error to "error" topic.
3. `controller.py` subscribes to "error" and publishes to "steer" topic.
4. `steer.py` subscribes to "steer_arduino" topic.
5. `throttle.py` gives constant throttle and publishes to "arduino" topic.

`serial_node_steer.py` and `serial_node_throttle.py` used for communication with arduino using rosserial.

`steer_joy.py` is to be used for joystick control only.

Download model folder from [here](https://drive.google.com/drive/folders/1-HP9OF_aj281cIZcciGRsn826bf1wL1B?usp=share_link).


## Setting up

1. Run `sudo apt-get install ros-noetic-usb-cam` to install camera node for video feed
2. Run `sudo apt-get install ros-noetic-joy` on terminal to install joystick package.
3. Run `rosrun joy joy_node` to start the node.
4. Run `rostopic echo joy` to see the topic result.
5. Run `rosrun usb_cam usb_cam_node` for the camera feed for logitech usb cam feed.
6. Run `roslaunch depthai_examples mobile_publisher.launch` for oak-d lite cam feed.

7. Run `python3 -m pip install tensorflow` on terminal to install tensorflow for `vision.py`.
8. Run `sudo apt-get install ros-noetic-rosserial-arduino` to install rosserial for communication with arduino.

# Setting up Oak-D Lite:

1. Follow steps from  [here](https://github.com/luxonis/depthai-ros).
2. Follow every step till Docker(excluded), then go to 'Executing a file'.
3. There a 8 packages , do not use catkin_build_isolated but use catkin build.
4. After build, simply launch file given above at point 6 in Setting up.
5. Multiple topics will appear and desired image will be published on some topic.

# Segmentation results

![Alt Text](wda_AdobeExpress.gif )

# Testing results

![Alt Text](aaa.gif )

