#!/usr/bin/env python3
import rospy
import csv
import os
from geometry_msgs.msg import PoseStamped


class PoseReader:
    def __init__(self):
        # Initialize the ROS node
        if not rospy.core.is_initialized():
            rospy.init_node("pose_reader", anonymous=True)

        # Path for the CSV file
        self.latest_pose = None

        # Subscribe to the /estimated_pose topic
        self.subscriber = rospy.Subscriber("/estimated_pose", PoseStamped, self.callback)

        self.header = ["time", "px", "py", "pz", "ox", "oy", "oz", "ow"]


    def callback(self, msg: PoseStamped):
        # Extract timestamp
        t = msg.header.stamp.to_sec()
        # Extract position
        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        # Extract orientation (quaternion)
        ox, oy, oz, ow = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

        self.latest_pose = [t, px, py, pz, ox, oy, oz, ow]
        # rospy.loginfo(f"Logged pose at t={t:.3f}: pos=({px:.2f},{py:.2f},{pz:.2f})")

if __name__ == "__main__":
    # Create an instance of the PoseLogger
    logger = PoseReader()

    # Keep the node running
    rospy.loginfo("Pose logger is running...")
    rospy.spin()