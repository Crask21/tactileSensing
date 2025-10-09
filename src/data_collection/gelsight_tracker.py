from src.data_collection.pose import PoseReader
from src.data_collection.triple_gelsight import TripleGelsight
from src.data_collection.tactile import TactileReader
import os
import time
import threading


class GelSightTracker:
    def __init__(self, config_path: str, save_path: str = None, class_name: str = None):
        # ------------------------ initialise data collectors ------------------------ #
        self.gelsight = TripleGelsight(config_path=config_path)
        self.pose_reader = PoseReader()
        self.tactile_reader = TactileReader()



        self.recording: bool = False
        self.record_filepath: str = None
        self.time = None
        self.record_frequency = 10  # Hz

        # --------------------- initial gelsight read to warm up --------------------- #
        self.gelsight.read_frames()

        # --------------------------- create file structure -------------------------- #
        self.save_path = save_path
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        self.set_class(class_name) if class_name else None
        # ------------------------------------- - ------------------------------------ #

        self.sample_no = len(os.listdir(self.tactile_path))
        print(f"Sample number: {self.sample_no}")

    def set_class(self, class_name: str):
        '''
        Sets up the directory structure for a new class of data.
        '''
        if not self.save_path:
            raise ValueError("Save path is not set.")
        self.class_path = os.path.join(self.save_path, class_name)
        os.makedirs(self.class_path, exist_ok=True)

        self.time_path = os.path.join(self.class_path, "time")
        self.tactile_path = os.path.join(self.class_path, "tactile")
        self.pose_path = os.path.join(self.class_path, "pose")
        self.videos_path = os.path.join(self.class_path, "videos")

        os.makedirs(self.time_path, exist_ok=True)
        os.makedirs(self.tactile_path, exist_ok=True)
        os.makedirs(self.pose_path, exist_ok=True)
        os.makedirs(self.videos_path, exist_ok=True)

    def read_all(self):
        images = self.gelsight.read_frames()
        pose = self.pose_reader.latest_pose
        tactile = self.tactile_reader.read() if self.tactile_reader.latest_data else None
        if self.recording:
            if pose is None:
                print("Warning: No pose data available.")
            if tactile is None:
                print("Warning: No tactile data available.")
            if any(img is None for img in images):
                print("Warning: One or more images not available.")

            # Save time
            t = time.time() - self.start_time
            with open(os.path.join(self.time_path, f"{self.sample_no:04d}.txt"), "a") as f:
                f.write(f"{t}\n")

            # Save pose
            if pose:
                with open(os.path.join(self.pose_path, f"{self.sample_no:04d}.csv"), "a") as f:
                    f.write(",".join(map(str, pose)) + "\n")

            # Save tactile
            if tactile:
                with open(os.path.join(self.tactile_path, f"{self.sample_no:04d}.csv"), "a") as f:
                    f.write(",".join(map(str, tactile)) + "\n")
            

        
        return images, pose, tactile

    def start_recording(self):
        if self.recording:
            print("Already recording.")
            return
        self.recording = True
        self.gelsight.start_recording(self.videos_path, self.sample_no)
        self.start_time = time.time()

        # Save time
        t = time.time() - self.start_time

        # Attempt to remove a 2 sec delay by pre-creating the files
        # open(os.path.join(self.time_path, f"{self.sample_no:04d}.txt"), "w")
        # open(os.path.join(self.pose_path, f"{self.sample_no:04d}.csv"), "w")
        # open(os.path.join(self.tactile_path, f"{self.sample_no:04d}.csv"), "w")

        # Start a separate thread for recording

        self.recording_thread = threading.Thread(target=self.record_thread, daemon=True)
        self.recording_thread.start()


        # print(f"Started recording sample {self.sample_no}.")

    def record_thread(self):

        while self.recording:
            next_time = time.time() + 1 / self.record_frequency
            self.read_all()
            sleep_duration = max(0, next_time - time.time())
            time.sleep(sleep_duration)

        
    def delete_last_sample(self):
        if self.recording:
            print("Cannot delete while recording.")
            return
        print("Deleting sample", self.sample_no - 1)
        if self.sample_no == 0:
            print("No samples to delete.")
        self.sample_no -= 1
        for path in [self.time_path, self.tactile_path, self.pose_path]:
            file_path = os.path.join(path, f"{self.sample_no:04d}.csv" if path != self.time_path else f"{self.sample_no:04d}.txt")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}.")
        video_dir = os.path.join(self.videos_path, f"{self.sample_no:04d}")
        if os.path.exists(video_dir):
            for root, dirs, files in os.walk(video_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(video_dir)
            print(f"Deleted {video_dir}.")
            

    def stop_recording(self):
        if not self.recording:
            print("Not currently recording.")
            return
        self.recording = False
        self.gelsight.stop_recording()
        print("frame count in sample ", self.sample_no, ":", self.gelsight.cam1.frame_count, self.gelsight.cam2.frame_count, self.gelsight.cam3.frame_count)
        # print(f"Stopped recording sample {self.sample_no}.")
        self.sample_no += 1
        self.recording_thread.join()


def unit_test():
    config_path = os.path.join(os.path.dirname(__file__), "triple_config.json")
    save_path = os.path.join(os.path.dirname(__file__), "data")
    tracker = GelSightTracker(config_path=config_path, save_path=save_path, class_name="test")
    
    # tracker.delete_last_sample()
    # q = input("Press Enter to start recording, or q to quit: ")
    # if q.lower() == 'q':
    #     return

    # time.sleep(3)
    tracker.start_recording()
    time.sleep(10)
    tracker.stop_recording()

    # tracker.start_recording()
    # time.sleep(5)
    # tracker.stop_recording()
    
    # tracker.start_recording()
    # time.sleep(5)
    # tracker.stop_recording()

def twonit_test():
    config_path = os.path.join(os.path.dirname(__file__), "triple_config.json")
    save_path = os.path.join(os.path.dirname(__file__), "data")
    tracker = GelSightTracker(config_path=config_path, save_path=save_path, class_name="test")
    

    tracker.gelsight.display_live()

if __name__ == "__main__":
    unit_test()
    # twonit_test()

