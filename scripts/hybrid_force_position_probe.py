from src.ur_controller.AutomateCapex import AutomateCapex
from src.utils.utils import recalibrate_gelsight_grasp_T
from src.ur_controller.utils import se3_to_pose
import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation as R


def hybrid_force_position_control(robot, direction_unit, stop_force, speed_scalar, acceleration, n):
    """
    Hybrid force-position control to maintain constant contact with the surface
    while rotating around the z-axis.

    Parameters:
    - robot: AutomateCapex instance
    - direction_unit: np.array, direction of force control (e.g., [0, 0, -1] for z-axis)
    - stop_force: float, desired force in the z-direction
    - speed_scalar: float, speed for position control
    - acceleration: float, acceleration for position control
    - n: int, number of probing points along the rotation
    """
    recalibrate_gelsight_grasp_T(robot=robot.robot)
    print(f"{robot.robot.T_tcp_gripper_tcp=}")

    # Desired GelSight pose in base frame
    # pose = [-0.19315984, -0.6893086, 0.165, 0.4 * np.cos(2 * np.pi * i / n), 0.4 * np.sin(2 * np.pi * i / n), np.pi]
    pose_start = [-0.19315984, -0.7893086, 0.165, 0,0, np.pi]
    rot = R.from_rotvec(pose_start[3:]).as_matrix()
    T = np.vstack([np.hstack([rot, np.array([pose_start[:3]]).T]), np.array([[0, 0, 0, 1]])])
    pose_start = sm.SE3(T)

    # Move to the initial position above the surface
    robot.robot.moveL(pose_start, 0.2, 0.2)

    # Activate hybrid force-position control
    # task_frame = se3_to_pose(robot.robot.T_tcp_gripper_tcp)
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 0, 0, 0]  # Position control for [x, y, rx, ry, rz], force control for z
    wrench = [0, 0, -stop_force, 0, 0, 0]  # Desired force in the z-direction
    force_type = 2  # Force control in the tool frame
    limits = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Motion limits

    robot.robot.ctrl.forceMode(task_frame, selection_vector, wrench, force_type, limits)
    # input("Press Enter to start rotation...")
    # Rotate around the z-axis while maintaining contact
    for i in range(n):

        pose = [-0.19315984, -0.7893086,   0.15, 0.4*np.cos(2*np.pi*i/n), 0.4*np.sin(2*np.pi*i/n), np.pi]
        rot = R.from_rotvec(pose[3:]).as_matrix()
        T = np.vstack([np.hstack([rot, np.array([pose[:3]]).T]), np.array([[0,0,0,1]])])
        pose = sm.SE3(T)
        
        robot.robot.moveL(pose, speed_scalar, acceleration)
    # for j in range(n):  # Adjust the number of steps for finer rotation
    #     delta_rot = sm.SE3.Rz(2 * np.pi / (n * 10))  # Small rotation step
    #     pose = pose * delta_rot
    #     robot.robot.moveL(pose, speed_scalar, acceleration)

    # Stop force control
    robot.robot.ctrl.forceModeStop()

    # Move back to the initial position above the surface
    robot.robot.moveL(pose_start, 0.2, 0.2)


if __name__ == "__main__":
    robot = AutomateCapex("192.168.1.130", 0)
    direction_unit = np.array([0, 0, -1])
    stop_force = 3
    speed_scalar = 0.1
    acceleration = 0.1
    n = 16

    hybrid_force_position_control(robot, direction_unit, stop_force, speed_scalar, acceleration, n)