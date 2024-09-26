import os
import numpy as np
import imageio.v2 as iio


def plot_animations_from_dir(dir_images):
        
    images = sorted(os.listdir(dir_images))
    if "animation.gif" in images:
            images.remove("animation.gif")

    frames = np.stack([iio.imread(f"{dir_images}/{img}") for img in images], axis=0)
    iio.mimwrite(f"{dir_images}/animation.gif", frames, format='GIF', duration=0.5, subrectangles=True)
    print(f"Animation saved in {dir_images}")




dirs_images_list = [

"/home/kalessia/flagAutomata/src/simulationAnalysis/sliding_puzzle_incremental_2024-09-26_19-44-22_two_bands_10x10/learning/run_000/plots/env/run_000_gen_004_individual_000/flag"

]

for dir_images in dirs_images_list:
    plot_animations_from_dir(dir_images)