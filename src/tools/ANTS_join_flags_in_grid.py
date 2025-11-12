import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# indexes_flags_to_plot = [0, 1, 2, 3, 5, 8, 13, 21, 34, 49]
# indexes_flags_to_plot = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
indexes_flags_to_plot = [1, 10, 20, 30, 40, 49]

crop_ratio = 0.1
n_rows = 1
n_cols = 7
images = []
labels = []

# Best inds modelA55 file paths
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-smile_16x16_modelA_4-[]-3_2-[5,5]-1_2025-10-21_01-54-53/run_005/plots/env/run_005_gen_01743_eval_0026146_individual_000/flag/component0/env0/plot_env_flag_run_005_gen_01743_eval_0026146_individual_000_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-SU_16x16_modelA_4-[]-3_2-[5,5]-1_2025-10-21_01-53-30/run_008/plots/env/run_008_gen_01783_eval_0026755_individual_009/flag/component0/env0/plot_env_flag_run_008_gen_01783_eval_0026755_individual_009_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_centered-half-discs_16x16_modelA_4-[]-3_2-[5,5]-1/run_000/plots/env/run_000_gen_01827_eval_0027408_individual_002/flag/component0/env0/plot_env_flag_run_000_gen_01827_eval_0027408_individual_002_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-french-cockade_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-57-47/run_012/plots/env/run_012_gen_01744_eval_0027911_individual_006/flag/component3/env0/plot_env_flag_run_012_gen_01744_eval_0027911_individual_006_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-italian-flag_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-56-46/run_008/plots/env/run_008_gen_01477_eval_0023641_individual_008/flag/component3/env0/plot_env_flag_run_008_gen_01477_eval_0023641_individual_008_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-rainbow-full_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-59-02/run_005/plots/env/run_005_gen_01468_eval_0023499_individual_010/flag/component3/env0/plot_env_flag_run_005_gen_01468_eval_0023499_individual_010_step_{idx:03}.png"

# Best inds modelC55 file paths
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-smile_16x16_modelC_4-[]-3_6-[5,5]-1_2025-10-21_01-34-38/run_005/plots/env/run_005_gen_01719_eval_0027505_individual_000/flag/component0/env0/plot_env_flag_run_005_gen_01719_eval_0027505_individual_000_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_bn-SU_16x16_modelC_4-[]-3_6-[5,5]-1_2025-10-21_01-31-22/run_015/plots/env/run_015_gen_01748_eval_0027984_individual_015/flag/component0/env0/plot_env_flag_run_015_gen_01748_eval_0027984_individual_015_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_centered-half-discs_16x16_modelC_4-[]-3_6-[5,5]-1/run_002/plots/env/run_002_gen_01639_eval_0026235_individual_010/flag/component0/env0/plot_env_flag_run_002_gen_01639_eval_0026235_individual_010_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-french-cockade_16x16_modelC_4-[]-3_6-[5,5]-3_2025-10-21_01-38-59/run_014/plots/env/run_014_gen_01630_eval_0027723_individual_012/flag/component3/env0/plot_env_flag_run_014_gen_01630_eval_0027723_individual_012_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-italian-flag_16x16_modelC_4-[]-3_6-[5,5]-3_2025-10-21_01-37-55/run_019/plots/env/run_019_gen_01623_eval_0027595_individual_003/flag/component3/env0/plot_env_flag_run_019_gen_01623_eval_0027595_individual_003_step_{idx:03}.png"
# filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-rainbow-full_16x16_modelC_4-[]-3_6-[5,5]-3_2025-10-21_01-41-02/run_003/plots/env/run_003_gen_01642_eval_0027918_individual_003/flag/component3/env0/plot_env_flag_run_003_gen_01642_eval_0027918_individual_003_step_{idx:03}.png"

target_filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-rainbow-full_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-59-02/plots_all_runs/plot_env0_flag_target_component3.png"
images.append(target_filepath)
labels.append(f"target")

for idx in indexes_flags_to_plot:
    filepath = f"/home/loi/flagAutomata/data_plots/ANTS/centered-half-discs_16x16_models-A-C-D-GECCO/sliding_puzzle_coordinates_2025-05-22_14-07-36_centered-half-discs_16x16/learning_coordinates_rgb-rainbow-full_16x16_modelA_4-[]-3_2-[5,5]-3_2025-10-21_01-59-02/run_005/plots/env/run_005_gen_01468_eval_0023499_individual_010/flag/component3/env0/plot_env_flag_run_005_gen_01468_eval_0023499_individual_010_step_{idx:03}.png"
    if os.path.exists(filepath):
        images.append(filepath)
        labels.append(f"step {idx}")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,3), dpi=600)
axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

for ax, img_path, label in zip(axes, images, labels):
    img = mpimg.imread(img_path)

    if crop_ratio > 0:
        h, w = img.shape[:2]
        y1, y2 = int(h*crop_ratio), int(h*(1-crop_ratio))
        x1, x2 = int(w*crop_ratio), int(w*(1-crop_ratio))
        img = img[y1:y2, x1:x2]

    ax.imshow(img)
    ax.axis("off")

    ax.text(
        0.5, -0.10, label,
        fontsize=20,
        ha='center', va='top',
        transform=ax.transAxes,
        color='black'
    )

for ax in axes[len(images):]:
    ax.axis("off")

plt.subplots_adjust(
    left=0.05,    # margine sinistro della figura
    right=0.95,   # margine destro della figura
    top=0.85,     # margine superiore (sotto titolo)
    bottom=0.1,   # margine inferiore (sopra eventuali label o bordo)
    wspace=0.01,   # spazio orizzontale tra i subplot (default 0.2)
    hspace=0.15    # spazio verticale tra i subplot (default 0.2)
)

# plt.suptitle("Best flag development over selected steps", fontsize=18, y=0.95)

plt.savefig("composed_flags_with_target.png", bbox_inches="tight")
plt.clf()
plt.close()

