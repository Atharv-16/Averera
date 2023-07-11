#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import numpy as np
import math
from math import *

joy_msg=Joy()
max_right_steer=21000
max_left_steer=19000
initial_brake=2000
full_brake=3000
normal_steer=20000
brake_scale_value=(full_brake-initial_brake)
steer_scale_value=(max_right_steer-normal_steer)

can_move=True
part_brakes=250
part_steer=100
max_throttle=400


global count
count=0
throttle_scale_value=max_throttle
max_throttle_scale=0
def steer(data):
    global steerr
    print(type(data))
    print("s"+str(data)[7]+str(data)[8]+str(data)[9]+str(data)[10]+str(data)[11])
    
   
    steerr="s"+str(data)[7]+str(data)[8]+str(data)[9]+str(data)[10]+str(data)[11]
    print(steerr)
    print("gnvgn")
    if(count==1):
        print("count")
        steer_pub.publish(steerr)
    else:
    	steer_pub.publish("s20000")


def joy_callback(data):
    joy_msg=data
    global count
    if(joy_msg.buttons[4]==1):
        count=1
        print("c1")
        #steer_pub.publish(steerr)
    else:
        
        count=0


    
        
        
            

if __name__ == '__main__':
    
    
    rospy.init_node('steer')
    pub= rospy.Publisher('arduino',String,queue_size=10)
    steer_pub=rospy.Publisher('steer_arduino',String,queue_size=10)
    rospy.Subscriber("steer",String,steer)
    rospy.Subscriber("joy",Joy,joy_callback)
    
    rate=rospy.Rate(10)
    rospy.spin()
