#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import numpy as np
import math
from math import *

joy_msg=Joy()


global count
count=0
def throttle(data):
    global count
    if(count==1):
        print("count")
        pub.publish("t300")
        pub.publish("b0")
    else:
    	pub.publish("t0")
    	pub.publish("b3000")


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
    rospy.Subscriber("steer",String,throttle)
    rospy.Subscriber("joy",Joy,joy_callback)
    
    rate=rospy.Rate(10)
    rospy.spin()
