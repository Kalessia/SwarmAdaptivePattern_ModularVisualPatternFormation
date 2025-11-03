import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def plot_colorbar(color_mode, orientation='vertical'):
    if color_mode == 'rgb':
        channels = ['R', 'G', 'B']
        colors = ['red', 'green', 'blue']

        for channel, color in zip(channels, colors):
            fig = plt.figure(figsize=(1, 6))
            cax = fig.add_axes([0.25, 0.05, 0.3, 0.9])  # [left, bottom, width, height] en fraction de figure
            
            cmap = mpl.colors.LinearSegmentedColormap.from_list(f"cmap_{channel}", [(0, 0, 0), mpl.colors.to_rgb(color)])
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            ticks = np.linspace(0,1,5)

            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{t:.1f}" for t in ticks])

            plt.savefig(f"colorbar_rgb_{channel}_{orientation}.png", dpi=300)
            plt.close()

    else: # Monochrome1 e Monochrome2
        fig = plt.figure(figsize=(1, 6))
        cax = fig.add_axes([0.25, 0.05, 0.3, 0.9])  # [left, bottom, width, height] en fraction de figure

        if color_mode == 'monochrome1':
            colors = [(0.0, 0.0, 0.0), (0.7, 0.9, 1.0)]  # black → light blue
            # cmap = mpl.colors.ListedColormap(np.linspace(colors[0], colors[1], 100))
            # norm = mpl.colors.Normalize(vmin=0, vmax=1)
            # ticks = np.linspace(0,1,5)

        elif color_mode == 'monochrome2':
            colors = [(0.0, 0.0, 0.0), (1.0, 0.4, 0.4)]  # black → light red
            # cmap = mpl.colors.LinearSegmentedColormap.from_list("black_red", colors) # LinearSegmentedColormap works from -1 to 1
            # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            # ticks = np.linspace(-1,1,5)

        else:
            print("Error: color_mode not recognized")
            return

        cmap = mpl.colors.ListedColormap(np.linspace(colors[0], colors[1], 100))
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        ticks = np.linspace(0,1,5)

        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}" for t in ticks])

        plt.savefig(f"colorbar_{color_mode}_{orientation}.png", dpi=300)
        plt.close()


plot_colorbar('monochrome1')
plot_colorbar('monochrome2')
plot_colorbar('rgb')



# @staticmethod
# def plot_flag(grid_nb_rows, grid_nb_cols, setup_name, run, nb_ind, gen, nb_eval, n, step, flag_list, fitness, env_id, permutated_pos=[], deleted_pos=[], nb_moves_per_step=0, analysis_dir_plots=None):

#     if isinstance(flag_list[0], (list, tuple)): # this means that we have a flag with N dimensions (called components)
#         flag_components = swarmGrid.get_flag_components(flag_list)

#         if len(flag_list[0]) == 3: # flag 3D, phenotype = [r,g,b] and r, g, b are floats in [0,1]
#             flag_list = flag_components + [flag_list]
#         else: # flag 2D, phenotype = [x,y] and x, y are floats in [0,1]
#             flag_list = flag_components
#     else: # flag 1D, phenotype = p and p is a float in [0,1]
#         flag_list = [flag_list]

#     for n_flag, flag in enumerate(flag_list):

#         # Color profile detection: 
#         first_elem_flag = flag[0]
#         is_rgb_flag = isinstance(first_elem_flag, (list, tuple)) and len(first_elem_flag) == 3
#         is_2d_or_1d_flag = (isinstance(first_elem_flag, float) or (isinstance(first_elem_flag, (list, tuple)) and len(first_elem_flag) == 2))

#         if is_rgb_flag:
#             color_mode = 'rgb'
#         elif is_2d_or_1d_flag:
#             if "signals" in analysis_dir_plots:
#                 color_mode = 'monochrome2'
#             else:
#                 color_mode = 'monochrome1'
#         else:
#             raise ValueError("Error in evironments.py, plot_flag. Unrecognized flag structure: expected float, 1D/2D list, or RGB triplet.")

#         fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

#         grid_pos = []
#         for row in range(grid_nb_rows):
#             for col in range(grid_nb_cols):
#                 grid_pos.append(tuple((row, col)))

#         if permutated_pos:
#             x_permutated = [pos[1] for pos in permutated_pos]
#             y_permutated = [-pos[0] for pos in permutated_pos]

#         if deleted_pos:
#             x_deleted = [pos[1] for pos in deleted_pos]
#             y_deleted = [-pos[0] for pos in deleted_pos]

#         if grid_nb_rows <= 10 and grid_nb_cols <= 10:
#             for row, col in [pos for pos in grid_pos if pos not in deleted_pos]:
#                 for neighbor_pos in [(row-1, col), (row, col-1), (row, col+1), (row+1, col)]:
#                     if swarmGrid.is_pos_valid(grid_nb_rows, grid_nb_cols, neighbor_pos) and neighbor_pos not in deleted_pos:
#                         ax.plot([col, neighbor_pos[1]], [-row, -neighbor_pos[0]], color='black', linestyle=':', zorder=1)
                    
#             circle_radius = 0.4


#         x = []
#         y = []
#         grey_values = []
#         for p, pos in enumerate(grid_pos):
#             grey_value = flag[p]

#             if grid_nb_rows > 10 or grid_nb_cols > 10:
#                 x.append(pos[1])
#                 y.append(-pos[0])
#                 grey_values.append(grey_value)

#             else:
#                 edgecolor_color = swarmGrid.map_color(grey_value, color_mode)
#                 facecolor_color = 'white'

#                 if color_mode != 'rgb' and grey_value > 0.9: # close to white
#                     circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=edgecolor_color, facecolor=facecolor_color, linestyle='--', linewidth=1.0, zorder=2)
#                 else:
#                     circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor=edgecolor_color, facecolor=facecolor_color, linewidth=6.0, zorder=2)

#                 if pos in permutated_pos:
#                     circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:green', facecolor=facecolor_color, linestyle='--', linewidth=2.0, zorder=2)    

#                 if pos in deleted_pos:
#                     circle = patches.Circle((pos[1], -pos[0]), circle_radius, edgecolor='tab:red', facecolor=facecolor_color, linestyle='--', linewidth=2.0, zorder=2)    
                
#                 ax.add_patch(circle)

#                 if grid_nb_rows < 6 and grid_nb_cols < 6:
#                     ax.text(pos[1], -pos[0], "(" + str(pos[0]) +"," + str(pos[1]) + ")\n"+ str(round(grey_value,2)), color='black', va='center', ha='center')
                
#                 # Hide axis lines and ticks, but still show labels
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
#                 ax.spines['left'].set_visible(False)
#                 ax.spines['bottom'].set_visible(False)


#         if grid_nb_rows > 10 or grid_nb_cols > 10:
#             if color_mode == 'monochrome1':
#                 colors = [(0.0, 0.0, 0.0), (0.7, 0.9, 1.0)]  # black → light blue
#                 cmap = ListedColormap(np.linspace(colors[0], colors[1], 100)) # ListedColormap works from 0 to 1
#                 plt.scatter(x, y, c=grey_values, cmap=cmap)
#             elif color_mode == 'monochrome2':
#                 colors = [(0.0, 0.0, 0.0), (1.0, 0.4, 0.4)]  # orange → black → light red
#                 cmap = matplotlib.colors.LinearSegmentedColormap.from_list("black_red", colors) # LinearSegmentedColormap works from -1 to 1
#                 norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
#                 plt.scatter(x, y, c=grey_values, cmap=cmap, norm=norm)
#             else:  # RGB
#                 plt.scatter(x, y, c=grey_values) # use RGB values (already in 0–1 range)
            
#             if permutated_pos:
#                 plt.scatter(x_permutated, y_permutated, c='tab:green') # agents permutated

#             if deleted_pos:
#                 plt.scatter(x_deleted, y_deleted, c='tab:red') # deleted agents
    

#         ax.set_aspect('equal')
#         plt.xlim(-0.5, grid_nb_cols-0.5)
#         plt.ylim(-grid_nb_rows+0.5, 0.5)
#         plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         # plt.xlabel(f"nb moves during this step = {nb_moves_per_step}", fontsize=12)


#         if setup_name:
#             # plt.title(f"Flag states - {setup_name}\nRun {run}, best individual {nb_ind}, step {step}.\nFlags distance = {fitness}", fontsize=10)
#             dir_name = f"{analysis_dir_plots}/{setup_name}/flag/component{n_flag}"
#             if not os.path.exists(dir_name):
#                 os.makedirs(dir_name, exist_ok=True)
#             plt.savefig(f"{dir_name}/{setup_name}_flag_run_{run:03}_best_ind_{nb_ind:03}_n_{n:03}_step_{step:03}.png")
#         else:
#             if nb_ind is not None:
#                 file_name = f"run_{run:03}_gen_{gen:05}_eval_{nb_eval:07}_individual_{nb_ind:03}"
#                 # plt.title(f"Flag states - learning.\nRun {run}, gen {gen}, nb_eval {nb_eval}, individual {nb_ind}, step {step}.\nFlags distance = {fitness}", fontsize=12)       
#                 dir_name = f"{analysis_dir_plots}/{file_name}/flag/component{n_flag}/env{env_id}"
#                 if not os.path.exists(dir_name):
#                     os.makedirs(dir_name, exist_ok=True)
#                 plt.savefig(f"{dir_name}/plot_env_flag_{file_name}_step_{step:03}.png")
#             else:
#                 # plt.suptitle(f"Flag target {grid_nb_rows}x{grid_nb_cols}", fontsize=12)
#                 plt.savefig(f"{analysis_dir_plots}/plot_env{env_id}_flag_target_component{n_flag}.png")

#         plt.clf()
#         plt.close()