import os

"""
This file should be used after gathering the raw images from the triple gelsight-mini configuration.
The depth scan/filter functions extract the matching images across all observations and performs the
depth calculation based on the provided depth reconstruction nn provided by gelsight.
"""

video_folder = r"./DataCollection/videos"
save_folder = r"./DataCollection/data"



def scan_folder_structure():

    open(r"./DataCollection/train.txt", "w")
    with open(r"./DataCollection/train.txt", "a") as f:
        for i, cls in enumerate(os.listdir(video_folder)):
            print(cls)
            if cls == "z_control":
                continue

            folder_path = os.path.join(video_folder, cls)

            for folder in os.listdir(folder_path):
                f.write(f"{cls}\\{folder}.mp4 150 {i}\n")





if __name__ == "__main__":
    scan_folder_structure()