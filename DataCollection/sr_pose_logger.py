#!/usr/bin/env python3
import rospy
import csv
import os
from geometry_msgs.msg import PoseStamped

# Path for the CSV file
CSV_FILE = os.path.expanduser("./pose_log.csv")

def callback(msg: PoseStamped):
    # Extract timestamp
    t = msg.header.stamp.to_sec()
    # Extract position
    px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
    # Extract orientation (quaternion)
    ox, oy, oz, ow = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

    # Append row to CSV
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([t, px, py, pz, ox, oy, oz, ow])

    rospy.loginfo(f"Logged pose at t={t:.3f}: pos=({px:.2f},{py:.2f},{pz:.2f})")

def listener():
    rospy.init_node("pose_logger", anonymous=True)

    # Write header if file doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "px", "py", "pz", "ox", "oy", "oz", "ow"])

    rospy.Subscriber("/estimated_pose", PoseStamped, callback)
    rospy.spin()

if __name__ == "__main__":
    listener()
