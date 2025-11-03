def plot_multi_flag_fitnesses_from_file(data_flag_dir, setup_name, run, analysis_dir_plots):

    data_flag_files = os.listdir(data_flag_dir)
    for data_flag_file in data_flag_files:
        dataset = pd.read_csv(data_flag_dir+"/"+data_flag_file)
        x = dataset['Step'].tolist()
        y = dataset['Flags_distance'].tolist()
        n = int(data_flag_file.split("n_")[1].split(".csv")[0])
        plt.plot(x, y, label=f"n_{n}")

    plt.ylim(0, 1) # 0 and 1 are respectively min and max values of flag distance
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Flags distance", fontsize=12)
    plt.title(f"Flags distance related to the flag development over steps. Run {run}\n{setup_name}, {len(data_flag_files)} repetitions", fontsize=12)
    plt.legend()

    dir_name = analysis_dir_plots+"/"+setup_name
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(f"{dir_name}/{setup_name}_flag_fitnesses_run_{run:03}.png")

    plt.clf()
    plt.close()


plot_multi_flag_fitnesses_from_file(data_flag_dir, setup_name, run, analysis_dir_plots)