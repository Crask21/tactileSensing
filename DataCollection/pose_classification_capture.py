import cv2
import numpy as np
from gsrobotics.config import ConfigModel
from gsrobotics.utilities.image_processing import (
    stack_label_above_image,
    apply_cmap,
    color_map_from_txt,
    normalize_array,
    trim_outliers,
)
from gsrobotics.utilities.reconstruction import Reconstruction3D
from gsrobotics.utilities.visualization import Visualize3D
from gsrobotics.utilities.gelsightmini import GelSightMini
from gsrobotics.utilities.logger import log_message
import time


def saveFile(filename, cam):
    try:
        b = cv2.imwrite(
            filename,
            cam.current_frame
        )
    except Exception as e:
        log_message(f"Failed to save frame: {e}")

def View3D(config: ConfigModel):

    # Initialize the camera stream.
    cam1 = GelSightMini(target_width=config.camera_width, target_height=config.camera_height)
    devices = cam1.get_device_list()
    log_message(f"Available camera devices: {devices}")
    # For testing, select device index 0 (adjust if needed).
    cam1.select_device(list(devices.keys())[1])

    cam2 = GelSightMini(target_width=config.camera_width, target_height=config.camera_height)
    cam2.select_device(list(devices.keys())[2])
    cam3 = GelSightMini(target_width=config.camera_width, target_height=config.camera_height)
    cam3.select_device(list(devices.keys())[0])
    
    
    dirname = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dirname, config.video_save_path) 
    os.makedirs(save_path, exist_ok=True)

    start_index = 36*4
    cam1.frame_count = start_index
    cam2.frame_count = start_index
    cam3.frame_count = start_index

    try:
        # --------------------------- COLLECT N_RECORDINGS --------------------------- #
        # while len([name for name in os.listdir(save_path)]) < config.n_recordings:
        log_message("\n----------------------------------")
        # Main loop: capture frames, compute depth map, and update the 3D view.
        print(f"Saving video to {save_path}")

    

        poses = ["up", "right", "down", "left"]
        for pose in poses:
            os.makedirs(os.path.join(os.path.join(save_path, pose),"device_1"), exist_ok=True)
            os.makedirs(os.path.join(os.path.join(save_path, pose),"device_2"), exist_ok=True)
            os.makedirs(os.path.join(os.path.join(save_path, pose),"device_3"), exist_ok=True)

        # --------------------------------- RECORDING -------------------------------- #
        while True:

            log_message(f"RECORDING number: {cam1.frame_count//4}")
            for pose in ["up", "right", "down", "left"]:
                input(f"Move to {pose}. Enter to continue")
                frames = [cam1.update_frame(dt=0, max_frame=config.n_frames), cam2.update_frame(dt=0, max_frame=config.n_frames), cam3.update_frame(dt=0, max_frame=config.n_frames)]
                if frames[0] is None or frames[1] is None or frames[2] is None:
                    continue
                saveFile(os.path.join(os.path.join(os.path.join(save_path, pose),"device_1"), f"frame_{cam1.frame_count//4:05d}.png"), cam1)
                saveFile(os.path.join(os.path.join(os.path.join(save_path, pose),"device_2"), f"frame_{cam2.frame_count//4:05d}.png"), cam2)
                saveFile(os.path.join(os.path.join(os.path.join(save_path, pose),"device_3"), f"frame_{cam3.frame_count//4:05d}.png"), cam3)
            if cam1.frame_count//4 >= config.n_recordings:
                break


            

    except KeyboardInterrupt:
        log_message("Exiting...")
    finally:
        # Release the camera and close windows.
        if cam1.camera is not None:
            cam1.camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    from gsrobotics.config import GSConfig
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Run the Gelsight Mini Viewer with an optional config file."
    )
    parser.add_argument(
        "--gs-config",
        type=str,
        default=None,
        help="Path to the JSON configuration file. If not provided, default config is used.",
    )

    args = parser.parse_args()

    if args.gs_config is not None:
        log_message(f"Provided config path: {args.gs_config}")
    else:
        log_message(f"Didn't provide custom config path.")
        log_message(
            f"Using default config path './default_config.json' if such file exists."
        )
        log_message(
            f"Using default_config variable from 'config.py' if './default_config.json' is not available"
        )
        args.gs_config = "default_config.json"

    gs_config = GSConfig(args.gs_config)
    View3D(config=gs_config.config)
