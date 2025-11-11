from src.ur_controller.AutomateCapex import AutomateCapex
from src.utils.utils import recalibrate_gelsight_grasp_T
import numpy as np
import spatialmath as sm
from scipy.spatial.transform import Rotation as R
from src.data_collection.gelsight_tracker import GelSightTracker 
from tqdm import tqdm
import os
import glob



def main():
    
    robot = AutomateCapex("192.168.1.130", 0)
    direction_unit = np.array([0, 0, -1])
    stop_force = 7
    max_distance = 0.1
    speed_scalar = 0.1
    acceleration = 0.1
    

    # Get reading from FT Sensor
    base_force = robot.robot.recv.getActualTCPForce()  # should return a 3-element list or array
    base_force = np.array(base_force[:3])

    # Project the force vector onto the direction unit vector using dot product
    force_direction = -np.dot(base_force, direction_unit)
    print(force_direction)

    # Initialize GelSight Tracker
    config_path = "src/data_collection/triple_config.json"
    save_path = "data"
    class_name = "triangle"
    tracker = GelSightTracker(config_path=config_path, save_path=save_path, class_name=class_name)

    while len(glob.glob(f"/home/user/py_ws/crask/tactileSensing/data/{class_name}/time/*")) < 50:
        sample = len(glob.glob(os.path.join(save_path, class_name, "time", "*")))

        # if sample % 10 == 0:
        q = input(f"Press Enter to start sample {sample},d to delete, or q to quit: ")
        if q.lower() == 'q':
            return
        elif q.lower() == 'd':
            tracker.delete_last_sample()
            continue

        # ---------------------------- One probe operation --------------------------- #
        recalibrate_gelsight_grasp_T(robot=robot.robot)


        n = 8
        for i in tqdm(range(n), desc="Probing"):
            # print(f"Probe: {i}/{n}")
            # 2) Desired GelSight pose in base frame: [0, -0.70, 0.20, 0, 0, pi]
            
            pose = [-0.19315984, -0.6893086,   0.165, 0.4*np.cos(2*np.pi*i/n), 0.4*np.sin(2*np.pi*i/n), np.pi]
            rot = R.from_rotvec(pose[3:]).as_matrix()
            T = np.vstack([np.hstack([rot, np.array([pose[:3]]).T]), np.array([[0,0,0,1]])])
            pose = sm.SE3(T)

            #Circle above:
            robot.robot.moveL(pose, 0.2, 0.2)
            
            if i == 0:
                tracker.start_recording()
            #Poke:
            robot.speed_until_force(direction_unit, stop_force, max_distance, speed_scalar, acceleration)
            
            #Circle above:
            robot.robot.moveL(pose, 0.2, 0.2)
        tracker.stop_recording()
        # --------------------------------- probe end -------------------------------- #
    
    

    



if __name__ == "__main__":
    main()