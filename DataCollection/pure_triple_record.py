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


def UpdateView(
    images: list[np.ndarray],
    cam_stream: GelSightMini,
    reconstruction: Reconstruction3D,
    visualizer3D: Visualize3D,
    cmap: np.ndarray,
    config: ConfigModel,
    window_title: str,
):

    # Apply labels above images
    frame_labeled = stack_label_above_image(
        images[0], f"Camera Feed {int(cam_stream.fps)} FPS", 30
    )
    frame2_labeled = stack_label_above_image(
        images[1], f"Camera Feed {int(cam_stream.fps)} FPS", 30
    )
    frame3_labeled = stack_label_above_image(
        images[2], f"Camera Feed {int(cam_stream.fps)} FPS", 30
    )
    # contact_mask_labeled = stack_label_above_image(contact_mask_rgb, "Contact Mask", 30)
    # Increase spacing between images by adding black spacers
    spacing_size = 30
    horizontal_spacer = np.zeros(
        (frame_labeled.shape[0], spacing_size, 3), dtype=np.uint8
    )

    top_row = np.hstack(
        (
            frame_labeled,
            horizontal_spacer,
            frame2_labeled,
            horizontal_spacer,
            frame3_labeled,
        )
    )

    display_frame = top_row

    # Scale the display frame
    display_frame = cv2.resize(
        display_frame,
        (
            int(display_frame.shape[1] * config.cv_image_stack_scale),
            int(display_frame.shape[0] * config.cv_image_stack_scale),
        ),
        interpolation=cv2.INTER_NEAREST,
    )
    display_frame = display_frame.astype(np.uint8)

    # Show the combined image.
    cv2.imshow(window_title, display_frame)


def View3D(config: ConfigModel):
    WINDOW_TITLE = "Multi-View (Depth 1, Depth 2, Depth 3)"

    reconstruction = Reconstruction3D(
        image_width=config.camera_width,
        image_height=config.camera_height,
        use_gpu=config.use_gpu,  # Change to True if you want to use CUDA.
    )

    # Load the trained network using the existing method in reconstruction.py.
    if reconstruction.load_nn(config.nn_model_path) is None:
        log_message("Failed to load model. Exiting.")
        return

    if config.pointcloud_enabled:
        # Initialize the 3D Visualizer.
        visualizer3D = Visualize3D(
            pointcloud_size_x=config.camera_width,
            pointcloud_size_y=config.camera_height,
            save_path="",  # Provide a path if you want to save point clouds.
            window_width=int(config.pointcloud_window_scale * config.camera_width),
            window_height=int(config.pointcloud_window_scale * config.camera_height),
        )
    else:
        visualizer3D = None

    cmap = color_map_from_txt(
        path=config.cmap_txt_path, is_bgr=config.cmap_in_BGR_format
    )

    # Initialize the camera stream.
    cam1 = GelSightMini(
        target_width=config.camera_width, target_height=config.camera_height
    )
    devices = cam1.get_device_list()
    log_message(f"Available camera devices: {devices}")
    # For testing, select device index 0 (adjust if needed).
    cam1.select_device(1)
    cam1.fps = 15  # Set FPS for the camera

    cam2 = GelSightMini(target_width=config.camera_width, target_height=config.camera_height)
    cam2.select_device(2)
    cam2.fps = 15  # Set FPS for the camera
    cam3 = GelSightMini(target_width=config.camera_width, target_height=config.camera_height)
    cam3.select_device(3)
    cam3.fps = 15  # Set FPS for the camera

    # Main loop: capture frames, compute depth map, and update the 3D view.
    dirname = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dirname, config.video_save_path) 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    path = f"{base}{timestamp}{ext}/"
    print(f"Saving video to {path}")
    print(os.path.isdir(path))
    # Create the directory for saving the video if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cam1.start_recording(filepath=path)
    cam2.start_recording(filepath=path)
    cam3.start_recording(filepath=path)


    import time
    t = time.time()

    # Create a window for displaying the images.
    # cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_TITLE, 640, 480)


    
    try:
        while True:
            # Get a new frame from the camera.
            # frame1 = cam1.update(dt=0)
            # frame2 = cam2.update(dt=0)
            # frame3 = cam3.update(dt=0)

            


            frames = [cam1.update(dt=0), cam2.update(dt=0), cam3.update(dt=0)]
            # print(frames[0].shape, frames[1].shape, frames[2].shape)
            if frames[0] is None or frames[1] is None or frames[2] is None:
                continue

            # Convert color
            frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

            UpdateView(
                images=frames,
                cam_stream=cam1,
                reconstruction=reconstruction,
                visualizer3D=visualizer3D,
                cmap=cmap,
                config=config,
                window_title=WINDOW_TITLE,
            )

            if time.time() - t > 10:
                break

            # Exit conditions.
            # When press 'q' on keyboard
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Check if the window has been closed by the user.
            # cv2.getWindowProperty returns a value < 1 when the window is closed.
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                # workaround to better catch widnow exit request
                for _ in range(5):
                    cv2.waitKey(1)
                break
        cam1.stop_recording()
        cam2.stop_recording()
        cam3.stop_recording()

        log_message(f"Video saved to {path}")

    except KeyboardInterrupt:
        log_message("Exiting...")
    finally:
        # Release the camera and close windows.
        if cam1.camera is not None:
            cam1.camera.release()
        cv2.destroyAllWindows()
        if visualizer3D:
            visualizer3D.visualizer.destroy_window()


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
