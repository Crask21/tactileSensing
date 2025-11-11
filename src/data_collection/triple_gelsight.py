from gsrobotics.utilities.gelsightmini import GelSightMini
from gsrobotics.config import ConfigModel, GSConfig
from gsrobotics.utilities.logger import log_message
from pydantic import BaseModel, ValidationError
from typing import List
import numpy as np
import os
import json
import cv2

class TripleGelsight:
    def __init__(self, config_path: str = None):
        self.config = GSConfig(config_path=config_path).config
        # print(self.config)
        cam1_id, cam2_id, cam3_id = self.load_id(config_path)
        
        # -------------------------- Select camera 1 via ID -------------------------- #
        self.cam1 = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height
        )
        cam_device_list = self.cam1.get_device_list()
        cam_ind = next((index for index, id in cam_device_list.items() if id[0] == cam1_id), None)
        self.cam1.select_device(cam_ind)
        
        # -------------------------- Select camera 2 via ID -------------------------- #
        # print("\n camera 2 \n")
        self.cam2 = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height
        )
        cam_ind = next((index for index, id in cam_device_list.items() if id[0] == cam2_id), None)
        self.cam2.select_device(cam_ind)

        # -------------------------- Select camera 3 via ID -------------------------- #
        self.cam3 = GelSightMini(
            target_width=self.config.camera_width,
            target_height=self.config.camera_height
        )
        cam_ind = next((index for index, id in cam_device_list.items() if id[0] == cam3_id), None)
        self.cam3.select_device(cam_ind)

    def read_frames(self) -> List[np.ndarray]:
        frames = [self.cam1.update_frame(), self.cam2.update_frame(), self.cam3.update_frame()]
        if not all([frame is not None for frame in frames]):
            print([frame is not None for frame in frames])
            raise RuntimeError("Failed to read from one or more cameras.")
        return frames
    
    def start_recording(self, save_path: str, sample_no: int):
        '''
        Starts recording frames from all three cameras and saves them to the specified directory.
        Each camera's frames are saved in separate subdirectories: sample_no/cam1, sample_no/cam2, sample_no/cam3.
        '''
        save_path = os.path.join(save_path, f"{sample_no:04d}")
        os.makedirs(save_path, exist_ok=True)
        cam1_path = os.path.join(save_path, "cam1") 
        cam2_path = os.path.join(save_path, "cam2")
        cam3_path = os.path.join(save_path, "cam3")
        os.makedirs(cam1_path, exist_ok=True)
        os.makedirs(cam2_path, exist_ok=True)
        os.makedirs(cam3_path, exist_ok=True)

        self.cam1.start_recording_frames_2(cam1_path)
        self.cam2.start_recording_frames_2(cam2_path)
        self.cam3.start_recording_frames_2(cam3_path)

    def stop_recording(self):
        self.cam1.stop_recording()
        self.cam2.stop_recording()
        self.cam3.stop_recording()

    def load_id(self, config_path: str):
        """
        Loads id configuration from the specifiied JSON file, and saves the three
        specific camera ids to self.cam1_id, self.cam2_id, self.cam3_id.
        """
        if config_path is None:
            log_message(f"No config path. ERROR")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            cam1_id = data["cam1_id"]
            cam2_id = data["cam2_id"]
            cam3_id = data["cam3_id"]
            

        except (json.JSONDecoder, ValidationError, KeyError) as e:
            log_message(
                f"Warning: Invalid config file. Using default configuration. Error: {e}"
            )
            cam1_id = 0
            cam2_id = 1
            cam3_id = 2
        
        return cam1_id, cam2_id, cam3_id

    def display_live(self):
        """
        Displays live video feed from the three cameras in separate windows.
        Press 'q' to quit the live feed.
        """

        try:
            while True:
                frames = self.read_frames()
                cv2.imshow("Camera 1", frames[0])
                cv2.imshow("Camera 2", frames[1])
                cv2.imshow("Camera 3", frames[2])

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()
            self.stop_recording()

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "triple_config.json")
    gelsight = TripleGelsight(config_path=config_path)
    gelsight.display_live()
    