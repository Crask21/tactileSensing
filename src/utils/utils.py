import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import rospy
from geometry_msgs.msg import PoseStamped
import rtde_receive
from src.ur_controller.robot import Robot
import spatialmath as sm

def pose_to_matrix(pose: np.array) -> np.array:
    """Convert geometry_msgs/Pose to 4x4 homogeneous matrix."""
    t = pose.position
    q = pose.orientation

    trans = np.array([t.x, t.y, t.z])
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    T = np.eye(4)
    T[:3,:3] = rot
    T[:3, 3] = trans
    return T

def tcp_to_matrix(tcp: np.array) -> np.array:
    """Convert UR RTDE TCP pose [x,y,z,rx,ry,rz] to 4x4 homogeneous matrix."""
    trans = np.array(tcp[:3])
    rotvec = np.array(tcp[3:])
    rot = R.from_rotvec(rotvec).as_matrix()

    T = np.eye(4)
    T[:3,:3] = rot
    T[:3, 3] = trans
    return T

def recalibrate_gelsight_grasp_T(robot: Robot) -> np.array:
    if not rospy.core.is_initialized():
        rospy.init_node("listener")
    #Move arm so camera has good view of gelsight handle
    robot.ctrl.moveL([-0.27012050067604276, -0.38655268021002365, 0.5045680326907392, 1.5437071890880918, -0.01577808098472903, -0.08004113081502745], 0.2, 0.2)
    print("Movement finished")
    time.sleep(1)
    message = rospy.wait_for_message("/estimated_pose", PoseStamped)
    
    TCP = robot.recv.getActualTCPPose()

    T_base_to_gs = pose_to_matrix(message.pose)
    T_base_to_tcp = tcp_to_matrix(TCP)

    # Relative transform
    T_tcp_to_gs = np.linalg.inv(T_base_to_tcp) @ T_base_to_gs

    robot.T_tcp_gripper_tcp = sm.SE3(T_tcp_to_gs)