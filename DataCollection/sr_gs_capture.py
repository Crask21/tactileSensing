import cv2
import numpy as np
import time
import os

# Device IDs for the first stream of each GelSight
CAMERAS = ["/dev/video2", "/dev/video4", "/dev/video6"]

# Output folder
SAVE_DIR = r"./DataCollection/sh_gs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Open video capture objects
caps = []
for dev in CAMERAS:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    caps.append(cap)



n = 0
cls = ["z_control"]
print(f"Move to {cls[n]}")
for c in cls:
    os.makedirs(os.path.join(SAVE_DIR, c, "device_1"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, c, "device_2"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, c, "device_3"), exist_ok=True)

while True:
    frames = []
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Camera {i} failed to grab frame")
            frame = np.zeros((320, 240, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (320, 240))
        frames.append(frame)
    print(frame.shape)

    # Stack frames horizontally for display
    stacked = np.hstack(frames)
    cv2.imshow("GelSight Stream (224x224 each)", stacked)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC to exit
        break
    elif key == ord("s"):
        for i, frame in enumerate(frames):
            filename = os.path.join(SAVE_DIR,f"{cls[n%4]}", f"device_{i+1}",f"{n//4:05d}.png")
            # cv2.imwrite(filename, frame)
            # print(f"[INFO] Saved {filename}")
        n += 1
        print(f"Move to {cls[n%4]}")

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()