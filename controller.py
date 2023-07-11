#!/usr/bin/env python
#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32, String

latest_float = 0.0

def float_callback(data):
    # Store the latest received float data
    global latest_float
    latest_float = data.data

def string_publisher():
    rospy.init_node('controller', anonymous=True)
    pub = rospy.Publisher('steer', String, queue_size=10)
    rospy.Subscriber('error', Float32, float_callback)
    
    rate = rospy.Rate(1)  # Set the publishing rate to 1 Hz
    
    while not rospy.is_shutdown():
        # Process the latest float data
        string_data = str(latest_float)
        # Publish the string data
        received_float=latest_float
        print(received_float)
        # Convert the float to string
        if(received_float>2 and received_float<10):
            pub.publish("20300")
        elif(received_float>10 and received_float<20):
            pub.publish("20500")
        elif(received_float>20 and received_float<30):
            pub.publish("20600")
        elif(received_float>30 and received_float<40):
            pub.publish("20700")
        elif(received_float>40 and received_float<50):
            pub.publish("20850")
        elif(received_float>50):
            pub.publish("21000")
        elif(received_float>-10 and received_float<-2):
            pub.publish("19300")
        elif(received_float>-20 and received_float<-10):
            pub.publish("19500")
        elif(received_float>-30 and received_float<-20):
            pub.publish("19400")
        elif(received_float>-40 and received_float<-30):
            pub.publish("19300")
        elif(received_float>-50 and received_float<-40):
            pub.publish("19150")
        elif(received_float<-50):
            pub.publish("19000")
        else:
            pub.publish("20000")
        # pub.publish(string_data)
        rate.sleep()

if __name__ == '__main__':
    try:
        string_publisher()
    except rospy.ROSInterruptException:
        pass









# import rospy
# from std_msgs.msg import Float32, String

# def float_callback(data):
#     # Process the received float data
#     received_float = data.data
#     print(received_float)
#     # Convert the float to string
#     if(received_float>2 and received_float<10):
#         pub.publish("20300")
#     elif(received_float>10 and received_float<20):
#         pub.publish("20500")
#     elif(received_float>20 and received_float<30):
#         pub.publish("20600")
#     elif(received_float>30 and received_float<40):
#         pub.publish("20700")
#     elif(received_float>40 and received_float<50):
#         pub.publish("20850")
#     elif(received_float>50):
#         pub.publish("21000")
#     elif(received_float>-10 and received_float<-2):
#         pub.publish("19300")
#     elif(received_float>-20 and received_float<-10):
#         pub.publish("19500")
#     elif(received_float>-30 and received_float<-20):
#         pub.publish("19400")
#     elif(received_float>-40 and received_float<-30):
#         pub.publish("19300")
#     elif(received_float>-50 and received_float<-40):
#         pub.publish("19150")
#     elif(received_float<-50):
#         pub.publish("19000")
#     else:
#         pub.publish("20000")
#     string_data = str(received_float)
#     # Publish the string data
    

# def string_publisher():
#     rospy.init_node('string_publisher_node', anonymous=True)
#     global pub
#     pub = rospy.Publisher('steer', String, queue_size=10)
#     rospy.Subscriber('error', Float32, float_callback)
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         string_publisher()
#     except rospy.ROSInterruptException:
#         pass























# #!/usr/bin/env python
# import rospy
# from geometry_msgs.msg import *
# from nav_msgs.msg import Odometry, Path
# from std_msgs.msg import Int16, String, Bool, Float32
# from collections import *
# import numpy as np
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from geometry_msgs.msg import Point
# import  math


# class Controller:
#     def __init__(self):
#         args_lateral_dict = {}
#         args_lateral_dict['K_P'] = 0.01
#         args_lateral_dict['K_I'] = 0.0
#         args_lateral_dict['K_D'] = 0.0

        
        
#         self._twist_msg = Twist()


#         #SUBCRIBERS
#         # self._odometry_subscriber = rospy.Subscriber("uwb_odom_low_pass", Odometry, self.odometry_cb)
#         self.error = rospy.Subscriber("error", Float32, self.error_y)

       
#         #PUBLISHERS
#         self._steering_command_publisher = rospy.Publisher("/steer", String, queue_size=10)

#         self._vehicle_controller = VehiclePIDController(
#             self, args_lateral=args_lateral_dict)

#     def error_y(self, err_y):
#         # with self.data_lock:
#         self._current_error = err_y
        
#         # self._current_speed = math.sqrt(odometry_msg.twist.twist.linear.x * 2 + odometry_msg.twist.twist.linear.y * 2 + odometry_msg.twist.twist.linear.z ** 2) * 3.6
        
#         self.run_step()
#         # rospy.loginfo(self._current_speed)
#         # print(odometry_msg.twist.twist.linear.x )
    

#     def run_step(self):
#         if not self._current_error:
#             rospy.loginfo("Waiting for our error")
#             return

        
#         rospy.loginfo("test")
        
#         if self._current_error!= 0:
#             steering = self._vehicle_controller.run_step(self._current_error)
#             self._twist_msg.angular.z = steering
#             self._steering_command_publisher.publish(str(steering))

#         else:
#             steering = 0
#             self._twist_msg.angular.z = steering
#             self._steering_command_publisher.publish(str(steering))
        




# class VehiclePIDController(object):

#     def __init__(self, node, args_lateral=None):

#         self.node = node
        
#         self._lat_controller = PIDLateralController(**args_lateral)



#     def run_step(self, error):

        
#         steering = self._lat_controller.run_step(error)
#         # control.steer = -steering
#         # control.throttle = throttle
#         # control.brake = 0.0
#         # control.hand_brake = False
#         # control.manual_gear_shift = False

#         return steering



# class PIDLateralController(object): 

#     def __init__(self, K_P=0.01, K_D=0.01, K_I=0.0):
#         self._K_P = K_P
#         self._K_D = K_D
#         self._K_I = K_I

#         rospy.loginfo("LATERAL : ")
#         rospy.loginfo(self._K_P)
#         rospy.loginfo(self._K_I)
#         rospy.loginfo(self._K_D)

#         self._e_buffer = deque(maxlen=10)
#         self.error = 0.0
#         self.error_integral = 0.0
#         self.error_derivative = 0.0

#     def run_step(self, error):
        

#         previous_error = self.error
#         self.error = error.data
#         # restrict integral term to avoid integral windup
#         self.error_integral = np.clip((self.error_integral) + self.error, -400.0, 400.0)
#         self.error_derivative = self.error - previous_error
#         output = self._K_P * self.error + self._K_I * self.error_integral+ self._K_D * self.error_derivative
#         # output=np.clip(output,-40,40)/40
#         print(output)

#         output=np.clip(output*5000 + 20000,19000,21000)   
#         rospy.loginfo(f"steer: {output}")
#         return (output)




# def main(args=None):
#     rospy.init_node("controller")
#     controller = Controller()
#     rospy.spin()

# if __name__ == "__main__":
#     main()
