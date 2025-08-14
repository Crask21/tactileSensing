import os
import cv2
from gsrobotics.utilities.reconstruction import Reconstruction3D
from gsrobotics.config import ConfigModel
from gsrobotics.config import GSConfig
from gsrobotics.utilities.logger import log_message
import numpy as np

"""
This file should be used after gathering the raw images from the triple gelsight-mini configuration.
The depth scan/filter functions extract the matching images across all observations and performs the
depth calculation based on the provided depth reconstruction nn provided by gelsight.
"""

video_folder = r"./DataCollection/pose_classification"
save_folder = r"./DataCollection/pose_processed"
# os.makedirs(save_folder, exist_ok=True)



def depth_scan():

    config = GSConfig("./DataCollection/config.json").config
    
    reconstruction1 = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )
    if reconstruction1.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return
    
    reconstruction2 = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )
    if reconstruction2.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return
    
    reconstruction3 = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )
    if reconstruction3.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return
    
    for i in range(1,55):
        img1 = cv2.imread(os.path.abspath(os.path.join(video_folder,f"z_control/device_1/frame_{str(i).zfill(5)}.png")))
        img2 = cv2.imread(os.path.abspath(os.path.join(video_folder,f"z_control/device_2/frame_{str(i).zfill(5)}.png")))
        img3 = cv2.imread(os.path.abspath(os.path.join(video_folder,f"z_control/device_3/frame_{str(i).zfill(5)}.png")))
        _,_,_,_ = reconstruction1.get_depthmap(image=img1, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)
        _,_,_,_ = reconstruction2.get_depthmap(image=img2, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)
        _,_,_,_ = reconstruction3.get_depthmap(image=img3, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)


    def depth_filter(img1, img2, img3):
        depth1 , _ , _ , _ = reconstruction1.get_depthmap(image=img1, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)
        depth2 , _ , _ , _ = reconstruction2.get_depthmap(image=img2, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)
        depth3 , _ , _ , _ = reconstruction3.get_depthmap(image=img3, markers_threshold=(config.marker_mask_min, config.marker_mask_max),)
        # print(depth1.shape)
        
        
        return depth1, depth2, depth3

    for i, cls in enumerate(os.listdir(video_folder)):
        log_message(cls)
        if cls == "z_control":
        # if cls != "triangle":
            continue
        # os.mkdir(os.path.join(save_folder,cls), exist_ok=True)
        # Create the directory
        os.makedirs(os.path.join(save_folder,cls),exist_ok=True)
        # print(i, cls)
        folder_path = os.path.join(video_folder, cls)
        # print(folder_path)
        if len(os.listdir(folder_path)) > 3:
            for folder in os.listdir(folder_path):
                print(folder)
                os.makedirs(os.path.join(os.path.join(save_folder, cls),folder), exist_ok=True)
                # print(cls, folder)
                sequence_path = os.path.join(folder_path, folder)
                device1 = os.path.join(sequence_path, "device_1")
                device2 = os.path.join(sequence_path, "device_2")
                device3 = os.path.join(sequence_path, "device_3")
                # print(f"{device1=}")
                # print(f"{device2=}")
                # print(f"{device3=}")
                deviceList1 = os.listdir(device1)
                deviceList2 = os.listdir(device2)
                deviceList3 = os.listdir(device3)
                
                for frame in range(len(os.listdir(device1))):
                    img1 = cv2.imread(os.path.join(device1, deviceList1[frame]))
                    img2 = cv2.imread(os.path.join(device2, deviceList2[frame]))
                    img3 = cv2.imread(os.path.join(device3, deviceList3[frame]))


                    try:
                        depth1, depth2, depth3 = depth_filter(img1, img2, img3)
                    except:
                        print(type(img1), type(img2), type(img3))
                    stacked_depth = np.stack([depth1, depth2, depth3], axis=-1)  # shape: (320, 240, 3)
                    stacked_depth = np.clip(stacked_depth, -7, 3)
                    stacked_depth = ((stacked_depth + 7) / 10 * 255).astype(np.uint8)
                    # print(stacked_depth.shape)
                    # Save the depth maps
                    cv2.imwrite(os.path.join(save_folder, cls, folder, f"frame_{str(frame).zfill(5)}.png"), stacked_depth)
        else:
            sequence_path = folder_path
            device1 = os.path.join(sequence_path, "device_1")
            device2 = os.path.join(sequence_path, "device_2")
            device3 = os.path.join(sequence_path, "device_3")

            deviceList1 = os.listdir(device1)
            deviceList2 = os.listdir(device2)
            deviceList3 = os.listdir(device3)
            
            for frame in range(len(os.listdir(device1))):
                img1 = cv2.imread(os.path.join(device1, deviceList1[frame]))
                img2 = cv2.imread(os.path.join(device2, deviceList2[frame]))
                img3 = cv2.imread(os.path.join(device3, deviceList3[frame]))


                try:
                    depth1, depth2, depth3 = depth_filter(img1, img2, img3)
                except:
                    print(type(img1), type(img2), type(img3))
                stacked_depth = np.stack([depth1, depth2, depth3], axis=-1)  # shape: (320, 240, 3)
                stacked_depth = np.clip(stacked_depth, -7, 3)
                stacked_depth = ((stacked_depth + 7) / 10 * 255).astype(np.uint8)
                # print(stacked_depth.shape)
                # Save the depth maps
                cv2.imwrite(os.path.join(save_folder, cls, f"frame_{str(frame).zfill(5)}.png"), stacked_depth)






if __name__ == "__main__":
    depth_scan()