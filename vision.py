import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rospy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import warnings
from std_msgs.msg import String,Float32
from sensor_msgs.msg import Image
warnings.warn("ignore")
import tensorflow as tf
import time



print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Define the DeepLabv3+ architecture
def create_deeplabv3plus(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    last_conv_layer = base_model.get_layer('block13_sepconv2_bn')

    # Create the atrous spatial pyramid pooling (ASPP) module
    x = last_conv_layer.output
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.UpSampling2D(size = (4, 4), interpolation = 'bilinear')(x)
    x = tf.keras.layers.UpSampling2D(size = (4, 4), interpolation = 'bilinear')(x)

    # Create the DeepLabv3+ model
    model = Model(inputs=base_model.input, outputs=x)

    return model

# Define input shape and number of classes
input_shape = (256, 256, 3)
loaded_model = create_deeplabv3plus(input_shape)
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="binary_crossentropy",
                  metrics="accuracy")
# loaded_model.load_weights("/kaggle/working/deeplabv3.h5")

from cv_bridge import CvBridge
img=Image()
frames = []
# print("RRRRRRRRRRRRRRRRRRRRRRR")
# loaded_model = build_unet_model()
# print("4324k234")
# loaded_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                 loss="binary_crossentropy",
#                 metrics="accuracy")
print("dwiuhdihaihi93")
loaded_model.load_weights("/home/atharv/Autonomy/test_vehicle_ws/src/test_vehicle/src/w.h5")
# told = 0
# imge=cv2.imread("/home/atharv/Autonomy/test_vehicle_ws/src/test_vehicle/src/a.png")
# print(imge)
# imge = cv2.resize(imge,(128,128))
    
# imge = imge/255    #clear
    
# imge = np.expand_dims(imge,axis = 0)
# ypr = loaded_model.predict(imge)

bridge = CvBridge()
latest_image = None
print("wda")
def image_callback(msg):
    global latest_image
    latest_image = msg
    print("dwadawdaw")

def main():
    global latest_image
    rospy.init_node('image_subscriber')
    rospy.Subscriber('/mobilenet_publisher/color/image', Image, image_callback)
    pub= rospy.Publisher('error',Float32,queue_size=1)

    while not rospy.is_shutdown():
        if latest_image is not None:
            # Convert ROS Image message to OpenCV image
            cv_image = bridge.imgmsg_to_cv2(latest_image, desired_encoding='bgr8')
            cv_image=cv2.resize(cv_image,(256,256))
            # Display the image
            # cv2.imshow('Image', cv_image)
            frame=cv_image
            f=frame
            
            #cv2.imshow("imagew",frame)
            frame = cv2.resize(frame,(256,256))
            
            frame = frame/255    #clear
            
            frame = np.expand_dims(frame,axis = 0)
            # cv2.imshow("imagew",f)
            # cv2.waitKey(0)
            o = loaded_model.predict(frame)
            
            # print(ypr.shape)

            o = np.squeeze(o)
            # o=np.asanyarray(o)
            # o = cv2.resize(o,(1024,1024))
            print(o.shape)
            ypr=o

            # print()

            cv2.imshow("output",o)

            #//////////////////////////////////////////////////////////////////////////////////
            # for i in range(0,len(ypr)):
            #     for j in range(0,len(ypr[i])):
            #         if ypr[i][j] >0.9:
            #             ypr[i][j] = 255
            #         else:
            #             ypr[i][j] = 0

            # copy = ypr[100:120,:]
            # mi = []
            # for i in range(0,len(copy)):
            #     a,b = 0,0
            #     for j in range(0,len(copy[i])):
            #         if copy[i][j] == 255:
            #             a = j
            #             break
            #     for k in range(len(copy[i])-1,0,-1):
            #         if copy[i][k] == 255:
            #             b = k
            #             break
            #     if a!= 0 and b!= 0:
            #         mid = (a+b)//2
            #         mi.append(mid)
            #         copy[i][mid] = 113
            #         copy[i][mid-1] = 113

            # m = pd.DataFrame({"COUNT":mi})
            # count = m["COUNT"].value_counts()
            # li = np.array(count[count  > 3].index)
            # if len(li) == 0:
            #     mean = 128
            # else:
            #     mean = int(li.mean())
            
            # img = frame.copy()
            # start_x1, start_y1, end_x1, end_y1 =  64, 0 , 64, 128
            # cv2.line(copy, (start_x1, start_y1),(end_x1,end_y1),(0),1)
            # start_x2, start_y2, end_x2, end_y2 = mean, 0 , mean, 128
            # cv2.line(copy, (start_x2, start_y2),(end_x2,end_y2),(0),1)
            # cv2.imshow("mid",copy)
                
            # frame = np.array(np.squeeze(frame), dtype = np.float32)
            # backtorgb = cv2.cvtColor(ypr,cv2.COLOR_GRAY2RGB)
                
            # weights = cv2.addWeighted(backtorgb,0.3,frame,0.7,0)
            # cv2.imshow("final",weights)
            # print(f"ERROR : {-start_x1 + start_x2}")
            # pub.publish(-start_x1+start_x2)
            # # image_message = bridge.cv2_to_imgmsg(copy, encoding="32FC1")
            # # epub.publish(image_message)
            # cv2.imshow("image",copy)
            # cv2.waitKey(20)
            # plt.imshow(weights)
            # plt.show()
            # cap.release()
            #cv2.destroyAllWindows()
            #////////////////////////////////////////////////////////////////////////////////////////////////////

            # num_channels = image.shape[2]

            # if num_channels == 1:
            #     print("The image is single-channel (grayscale).")
            # elif num_channels == 3:
            #     print("The image is multi-channel (color).")
            # else:
            #     print("The image has an unsupported number of channels.")


            # _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

            # # Display the thresholded image
            # cv2.imshow('Thresholded Image', thresholded_image)
            # cv2.waitKey(0)


#/////////////////////////////////////////////////////////////////////////////////////////mine
            kernel = np.array([[0, 5, 0],
                   [-1, 5, -1],
                   [0, 5, 0]])

    # Apply the kernel to the image
            sharpened_image = cv2.filter2D(o, -1, kernel)
            
            cv2.imshow("odutput",sharpened_image)

            ret, threshold = cv2.threshold(sharpened_image, 2, 255, cv2.THRESH_BINARY)
            threshold = np.uint8(threshold)
            cv2.imshow("thes",threshold)

            # Find contours in the binary image

            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            def find_array_with_same_first_element(known_array, arr):
                first_element = known_array[0]

                for sub_array in arr:
                    if np.array_equal(sub_array[0], first_element):
            # return sub_array
                        return sub_array

                # return None
            # canvas = np.zeros_like(threshold)

            # Draw contours on the canvas
            threshold = cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)
            
            # cv2.drawContours(threshold, contours, -1, (0, 255, 255), 10)
            # cv2.imshow('Image with Contours',threshold )
            

            if(len(contours)>0):
                # Find the largest contour based on contour area
                largest_contour = max(contours, key=cv2.contourArea)
                sorted_points = sorted(largest_contour[:, 0], key=lambda x: x[1])
                cv2.drawContours(threshold, largest_contour, -1, (0, 255, 255), 10)
                cv2.imshow('Image with Contours',threshold )
                grouped_points = {}
                for point in sorted_points:
                    y = point[1]
                    if y in grouped_points:
                        grouped_points[y].append(point)
                    else:
                        grouped_points[y] = [point]

                averages = []
                for y, points in grouped_points.items():
                    if len(points) == 2:
                        average_x = np.mean([point[0] for point in points])
                        averages.append((average_x, y))

                
                

                # Draw circles at the average points
                
                # Print the average points
                for point in averages:
                    print(f"Average point: ({point[0]}, {point[1]})")

                # image_color = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)
                for point in averages:
                    # print(point)
                    cv2.circle(threshold, (int(point[0]), int(point[1])), 5, (0,0, 255), 1)
                

                
                # Create a mask image of the same size as the original image
                # mask = np.zeros_like(sharpened_image)
                # # print(largest_contour)
                # h=256
                # for i in largest_contour:
                #     result = find_array_with_same_first_element(i, contours)
                #     # print(result)
                #     # if(c=0):
                # # print(h)
                # # Draw the largest contour on the mask image
                # cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

                # # Apply the mask to the original image
                # masked_image = cv2.bitwise_and(sharpened_image, mask)

    #         # Display the original image and the masked image
    #         # cv2.imshow('Original Image', image)
    #             cv2.imshow('Masked Image', masked_image)
    #             image_color = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)
    #             for point in averages:
    #                 #print(point)
    #                 cv2.circle(masked_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    #             cv2.imshow("color",masked_image)

    #             print(len(averages))
                if(len(averages)>0):
                    bottom_most_point = averages[-1]
                    print("Bottom-Most Average Point:", bottom_most_point)
                    cv2.circle(threshold, (int(bottom_most_point[0]), int(bottom_most_point[1])), 5, (255, 0, 0), -1)                

                
                h = 5  # Adjust the h value as desired

                # Find the first point, going from bottom to top, whose x-coordinate is greater than the bottom-most x-coordinate by h
                target_point_x = None
                target_point_y = None
                reversed_averages=averages[::-1]
                for point in reversed_averages:
                    x = point[0]
                    y = point[1]
                    if abs(x-bottom_most_point[0]) > h:
                        target_point_x = x
                        target_point_y = y
                        break

                # Print the x-coordinate of the target point
                if target_point_x is not None:
                    print("X-coordinate of Target Point:", target_point_x)
                    cv2.circle(threshold, (int(target_point_x), target_point_y), 5, (255, 0,0), -1)
                    # Define the two points
                    point1 = (int(bottom_most_point[0]),int(bottom_most_point[1]))  # (x1, y1)
                    point2 = (int(target_point_x),int(target_point_y))  # (x2, y2)

                    # Draw an arrowed line between the two points
                    color = (0, 0, 255)  # BGR color (red)
                    thickness = 2
                    cv2.arrowedLine(cv_image, point1, point2, color, thickness)
                    pub.publish(x-bottom_most_point[0])
                else:
                    print("No point found satisfying the condition.")
                    pub.publish(0)
                

                
                cv2.imshow("colored",threshold)
                cv2.imshow("coloreded",cv_image)
    #             h = 15  # Adjust the h value as desired

    #             # Find the x-coordinates of the nearest points within the range
    #             nearest_points_x = []
    #             for point in averages:
    #                 x = point[0]
    #                 y = point[1]
    #                 if bottom_most_point[0] - h <= x <= bottom_most_point[0] + h:
    #                     nearest_points_x.append(x)

    #             # Print the x-coordinates of the nearest points within the range
    #             # print("X-coordinates of Nearest Points:", nearest_points_x)
    #             # print(averages[-10])
    #             # print(nearest_points_x[-3])
    #             if(len(averages)>5):
    #                 pub.publish(averages[-5][0]-bottom_most_point[0])
    #             # # Convert the image to BGR color format
    #             # image_color = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    #             # # Draw circles at the average points
    #             # for point in averages:
    #             #     cv2.circle(image_color, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    #             # # return sharpened_image
        #//////////////////////////////////////////////////////////////////////////////////mine
            

        # Wait for a key press indefinitely
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    # rospy.spin()
    cv2.destroyAllWindows()

main()
