#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

CSV_FILE = os.path.expanduser("./pose_log.csv")

def animate(i):
    data = pd.read_csv(CSV_FILE)
    xs, ys, zs = data["px"].to_numpy(), data["py"].to_numpy(), data["pz"].to_numpy()

    ax.clear()
    ax.plot(xs, ys, zs, c="blue")
    ax.scatter(xs[-1], ys[-1], zs[-1], c="red", marker="o")  # latest point

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated Pose Trajectory")


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ani = animation.FuncAnimation(fig, animate, interval=500)  # refresh every 0.5s
    plt.show()
