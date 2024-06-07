import pickle
from simulation import simulate_trajectory_sample
import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({'font.size': 16,
                     'font.weight': 'bold'
                     })


def run_simulation_multi_parallel(t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, num_trajectories):
    # Create an iterable of arguments for simulate_trajectory
    args_list = [(t_cutoff, tau_A, tau_B, dt, lambda_A,
                  lambda_B, alpha)] * num_trajectories

    # Use multiprocessing Pool to parallelize the simulation
    with Pool() as pool:
        trajectories = list(
            tqdm.tqdm(pool.imap(simulate_trajectory_sample, args_list), total=num_trajectories))

    return trajectories


def analyze_trajectories_with_analytical(trajectories, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, analytical_msd_df, analytical_popul_df):
    max_time_steps = int(t_cutoff / dt) + 1
    msd_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}
    population_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}

    for traj in trajectories:
        initial_pos = np.array([traj['x'][0], traj['y'][0]])
        for t, (x, y, state) in enumerate(zip(traj['x'], traj['y'], traj['state'])):
            displacement = (x - initial_pos[0])**2 + (y - initial_pos[1])**2
            if state:  # Check if state is not empty
                msd_data[state][t] += displacement
                msd_data['overall'][t] += displacement
                population_data[state][t] += 1
                population_data['overall'][t] += 1

    time_steps = np.arange(max_time_steps) * dt
    plt.figure(figsize=(10, 6))
    x = analytical_msd_df['t']
    y_columns = ['MSD_A', 'MSD_B', 'MSD_overall']
    y_label = ['$MSD_A$', '$MSD_B$', '$MSD_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, analytical_msd_df[column], label=label, color=color)

    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        msd_data[state] /= population_data[state]
        plt.scatter(time_steps[1:], msd_data[state][1:]*2/10_000_000_000,
                    label=f'MSD: {state}', marker=marker, color=color)  # /len(trajectories)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    plt.xlim(min(time_steps), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$', fontweight='bold', fontsize=20)
    # plt.title(f"Mean Square Displacement over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    x1 = analytical_popul_df['t']
    y_columns = ['num_a', 'num_b', 'num_total']
    y_label = ['$N_A$', '$N_B$', '$N_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    # for column, label, color in zip(y_columns, y_label, colors):
    #    plt.plot(x1, analytical_popul_df[column], label=label, color=color)

    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        plt.plot(time_steps[1:], population_data[state][1:],
                 label=f'Population: {state}',  color=color)  # marker=marker,
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    plt.xlim(min(time_steps), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel('$N$', fontweight='bold', fontsize=20)
    # plt.title(f"Population over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()


def analyze_trajectories_with_analytical_bin(trajectories, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, analytical_msd_df, analytical_popul_df):
    max_time_steps = int(t_cutoff / dt) + 1
    msd_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}
    population_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}

    for traj in trajectories:
        initial_pos = np.array([traj['x'][0], traj['y'][0]])
        for t, (x, y, state) in enumerate(zip(traj['x'], traj['y'], traj['state'])):
            displacement = (x - initial_pos[0])**2 + (y - initial_pos[1])**2
            if state:  # Check if state is not empty
                msd_data[state][t] += displacement
                msd_data['overall'][t] += displacement
                population_data[state][t] += 1
                population_data['overall'][t] += 1

    time_steps = np.arange(max_time_steps) * dt
    plt.figure(figsize=(10, 6))
    x = analytical_msd_df['t']
    y_columns = ['MSD_A', 'MSD_B', 'MSD_overall']
    y_label = ['$MSD_A$', '$MSD_B$', '$MSD_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, analytical_msd_df[column], label=label, color=color)

    # Logarithmic binning of simulation data
    bin_edges = np.logspace(
        np.log10(min(time_steps[1:])), np.log10(max(time_steps)), 20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for state in ['A', 'B', 'overall']:
        msd_data[state] /= population_data[state]

    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        msd_data_state = msd_data[state][1:]*2 / 10_000_000_000
        valid_indices = ~np.isnan(msd_data_state)  # Find valid indices

        # Filter both arrays synchronously
        msd_data_state = msd_data_state[valid_indices]
        time_steps_filtered = time_steps[1:][valid_indices]

        bin_means, _, _ = stats.binned_statistic(
            time_steps_filtered, msd_data_state, statistic='mean', bins=bin_edges)
        plt.scatter(bin_centers, bin_means,
                    label=f'MSD: {state}', marker=marker, color=color)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(time_steps[1:]), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$', fontweight='bold', fontsize=20)
    # plt.title(f"Mean Square Displacement over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'orange']
    x1 = analytical_popul_df['t']
    y_columns = ['num_a', 'num_b', 'num_total']
    y_label = ['$N_A$', '$N_B$', '$N_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x1, analytical_popul_df[column], label=label, color=color)

    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        bin_means, _, _ = stats.binned_statistic(
            time_steps[1:], population_data[state][1:]/1000, statistic='mean', bins=bin_edges)
        plt.scatter(bin_centers, bin_means,
                    label=f'Population: {state}', marker=marker, color=color)
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(0.0001,1)
    plt.xlim(min(time_steps[1:]), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel('$N$', fontweight='bold', fontsize=20)
    # plt.title(f"Population over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()


def analyze_trajectories_bin(trajectories, t_cutoff, tau_A, tau_B, dt, lambda_A, lambda_B, alpha, analytical_msd_df, bin=20):
    max_time_steps = int(t_cutoff / dt) + 1
    msd_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}
    population_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}

    for traj in trajectories:
        initial_pos = np.array([traj['x'][0], traj['y'][0]])
        for t, (x, y, state) in enumerate(zip(traj['x'], traj['y'], traj['state'])):
            displacement = (x - initial_pos[0])**2 + (y - initial_pos[1])**2
            if state:  # Check if state is not empty
                msd_data[state][t] += displacement
                msd_data['overall'][t] += displacement
                population_data[state][t] += 1
                population_data['overall'][t] += 1

    time_steps = np.arange(max_time_steps) * dt
    plt.figure(figsize=(10, 6))
    x = analytical_msd_df['t']
    y_columns = ['MSD_A', 'MSD_B', 'MSD_overall']
    y_label = ['$MSD_A$', '$MSD_B$', '$MSD_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    # for column, label, color in zip(y_columns, y_label, colors):
    #    plt.plot(x, analytical_msd_df[column], label=label, color=color)

    # Logarithmic binning of simulation data
    bin_edges = np.logspace(
        np.log10(min(time_steps[1:])), np.log10(max(time_steps)), bin)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for state in ['A', 'B', 'overall']:
        msd_data[state] /= population_data[state]

    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        msd_data_state = msd_data[state][1:] / 1_000_000_000
        valid_indices = ~np.isnan(msd_data_state)  # Find valid indices

        # Filter both arrays synchronously
        msd_data_state = msd_data_state[valid_indices]
        time_steps_filtered = time_steps[1:][valid_indices]

        bin_means, _, _ = stats.binned_statistic(
            time_steps_filtered, msd_data_state, statistic='mean', bins=bin_edges)
        plt.plot(bin_centers, bin_means, label=f'MSD: {state}', color=color)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(time_steps[1:]), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$', fontweight='bold', fontsize=20)
    # plt.title(f"Mean Square Displacement over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'orange']
    for state, color, marker in zip(['A', 'B', 'overall'], colors, markers):
        bin_means, _, _ = stats.binned_statistic(
            time_steps[1:], population_data[state][1:], statistic='mean', bins=bin_edges)
        plt.plot(bin_centers, bin_means,
                 label=f'Population: {state}', color=color)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(min(time_steps[1:]), max(time_steps))
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel('$N$', fontweight='bold', fontsize=20)
    # plt.title(f"Population over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()


def save_load_traj(traj, filename, load_file=False):
    """Saves or loads trajectory data using pickle.

    Args:
        traj: The trajectory data (list of dictionaries).
        filename: The base filename (without '.pkl' extension).
        load_file: If True, loads the data; otherwise, saves it.

    Returns:
        The loaded trajectories if load_file is True, otherwise None (or True/False
        to indicate save success/failure).
    """

    if load_file:
        try:
            with open(f'{filename}.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Error: File '{filename}.pkl' not found.")
            return None

    else:
        try:
            with open(f'{filename}.pkl', 'wb') as f:
                pickle.dump(traj, f)
            return True  # Indicate save success
        except IOError:
            print(f"Error: Could not save data to '{filename}.pkl'.")
            return False


def import_analytical_num_df(csv_path):
    num_column_mapping = {
        '0': 't',
        '1': 'num_a',
        '2': 'num_b',
        '3': 'num_total',
        '4': 'shortN1',
        '5': 'shortN2',
        '6': 'shortN12',
        '7': 'LongLimN1',
        '8': 'LongLimN2',
        '9': 'LongLimN12',
    }
    # ,usecols=column_mapping.keys())
    df1_ = pd.read_csv(f'{csv_path}.csv', names=num_column_mapping.keys())
    # df2_ = pd.read_csv(f'/content/drive/MyDrive/Skoltech_Researches/Exciton_Semiconductors/Fig5L33.csv', names=column_mapping.keys())
    # df3_ = pd.read_csv(f'/content/drive/MyDrive/Skoltech_Researches/Exciton_Semiconductors/Fig5L4.csv', names=column_mapping.keys())
    # Rename columns using the mapping
    df1 = df1_.rename(columns=num_column_mapping)
    # df2 = df2_.rename(columns=column_mapping)
    # df3 = df3_.rename(columns=column_mapping)

    # df = pd.concat([df1, df2, df3], ignore_index=True)
    df1.head()
    return df1


def import_analytical_msd_df(csv_path):

    column_mapping = {
        '0': 't',
        '1': 'MSD_A',
        '2': 'MSD_B',
        '3': 'MSD_overall',
        '4': 'shortLim',
        '5': 'shortLim2',
        '6': 'MSDLongLim',
        #    '7': 'precision'
    }
    # ,usecols=column_mapping.keys())
    df1_ = pd.read_csv(f'{csv_path}.csv', names=column_mapping.keys())
    # df2_ = pd.read_csv(f'/content/drive/MyDrive/Skoltech_Researches/Exciton_Semiconductors/Fig5L33.csv', names=column_mapping.keys())
    # df3_ = pd.read_csv(f'/content/drive/MyDrive/Skoltech_Researches/Exciton_Semiconductors/Fig5L4.csv', names=column_mapping.keys())
    # Rename columns using the mapping
    df1 = df1_.rename(columns=column_mapping)
    # df2 = df2_.rename(columns=column_mapping)
    # df3 = df3_.rename(columns=column_mapping)

    # df = pd.concat([df1, df2, df3], ignore_index=True)
    df1.head()
    return df1


def plot_analytical_population(df1, savepath):
    x = df1['t']

    # Extract other columns for y-axis
    y_columns = ['num_a', 'num_b', 'num_total']
    y_label = ['$N_A$', '$N_B$', '$N_{overall}$']
    colors = ['red', 'blue', 'orange']

    # Plot all other columns against 't'
    plt.figure(figsize=(10, 6))
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, df1[column], label=label, color=color)
    # plt.plot(x, 0.25*df1['shorttime_limit1'], linestyle='--', color='black')

    # Add labels and legend
    plt.xlabel('$t,(ns)$')
    plt.ylabel('$N(t)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    # plt.title('Population over Time')
    plt.legend()
    plt.savefig(f'{savepath}.png')
    plt.show()


def plot_analytical_msd(df1, savepath):
    x = df1['t']

    # Extract other columns for y-axis
    y_columns = ['MSD_A', 'MSD_B', 'MSD_overall']
    y_label = ['$MSD_A$', '$MSD_B$', '$MSD_{overall}$']
    colors = ['red', 'blue', 'orange']

    # Plot all other columns against 't'
    plt.figure(figsize=(10, 6))
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, df1[column], label=label, color=color, linewidth=2)
    # plt.plot(x, df1['shorttime_limit2'], linestyle='--', color='black')
    plt.plot(x, df1['shorttime_limit3'], linestyle='--', color='black')
    plt.plot(x, df1['longtime_limit'], linestyle='--', color='black')

    # Add labels and legend
    plt.xlabel('$t, ns$', fontweight='bold', fontsize=20)
    plt.ylabel('$\\langle r^2(t) \\rangle, \\mu m^2$',
               fontweight='bold', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    # plt.title('Mean Square Displacement over Time')
    # plt.legend()
    plt.savefig(f'{savepath}.png')
    plt.show()


def plot_analytical_msd_single(df1, df2=None, df3=None, savepath=None):
    plt.figure(figsize=(10, 6))

    plt.plot(df1['t'], df1['MSD_A'], color='red', linewidth=2)
    if df2 is not None:
        plt.plot(df2['t'], df2['MSD_A'], color='blue', linewidth=2)
    if df3 is not None:
        plt.plot(df3['t'], df3['MSD_A'], color='orange', linewidth=2)

    plt.xlabel('$t$ (ns)', fontweight='bold', fontsize=14)
    plt.ylabel('$\\langle r^2(t) \\rangle$, $\\mu$m$^2$',
               fontweight='bold', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    plt.ylim(bottom=1e-3)
    # plt.grid(alpha=0.4)
    # plt.legend()
    if savepath:
        plt.savefig(savepath)
    plt.show()

# vectorised simulation plot


def plot_msd(result, ns=10_000_000_000):

    r = ((result['msd']['x'][:, 1:] - result['msd']['x'][:, 0].reshape(-1, 1))**2
         + (result['msd']['y'][:, 1:] - result['msd']['y'][:, 0].reshape(-1, 1))**2)

    t = result["msd"]['t'][1:]
    msd = (np.nansum(r, axis=0) / np.count_nonzero(np.isfinite(r), axis=0))

    msd_fast = np.zeros(result["msd"]["status"].shape[1] - 1)
    msd_slow = np.zeros(result["msd"]["status"].shape[1] - 1)

    for ii in range(msd_fast.size):
        try:
            msd_fast[ii] = np.mean(
                r[result["msd"]["status"][:, ii + 1] == 0, ii])
        except:
            msd_fast[ii] = np.inf

        try:
            msd_slow[ii] = np.mean(
                r[result["msd"]["status"][:, ii + 1] == 1, ii])
        except:
            msd_slow[ii] = np.inf

    plt.figure(figsize=(9, 6))
    plt.plot(t[1:], msd[1:]*2/ns, color="orange", label="all QP")
    plt.plot(t[1:], msd_slow[1:]*2/ns,
             color="blue", label="slow QP")
    plt.plot(t[1:], msd_fast[1:]*2/ns,
             color="red", label="fast QP")
    plt.xlim((0.001, 1000))
    # plt.plot(t[-10:], np.power(10, fit_coef[0] * np.log10(t[-10:])
    #                           + fit_coef[1]), 'k:', label=f"slope = {np.round(fit_coef[0], 2)}")

    # plt.legend(fontsize=14)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t, ns")
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$')
    # plt.grid(True)

# vectorised simulation plot


def plot_msd_with_analytical_bin(result, analytical_msd_df, num_bins=20, ns=10_000_000_000):
    r = ((result['msd']['x'][:, 1:] - result['msd']['x'][:, 0].reshape(-1, 1)) ** 2
         + (result['msd']['y'][:, 1:] - result['msd']['y'][:, 0].reshape(-1, 1)) ** 2)
    t = result["msd"]['t'][1:]  # Exclude the initial time point (t=0)

    msd = (np.nansum(r, axis=0) / np.count_nonzero(np.isfinite(r), axis=0))

    # MSD for each state
    msd_fast = np.zeros(t.size)
    msd_slow = np.zeros(t.size)

    for ii in range(t.size):
        msd_fast[ii] = np.mean(
            r[(result["msd"]["status"][:, ii + 1] == 0) & np.isfinite(r[:, ii]), ii])
        msd_slow[ii] = np.mean(
            r[(result["msd"]["status"][:, ii + 1] == 1) & np.isfinite(r[:, ii]), ii])

    # Logarithmic Binning for Simulation Data
    bin_edges = np.logspace(np.log10(min(t)), np.log10(max(t)), num_bins + 1)

    # Binned Mean Calculation with NaN Filtering
    binned_msd, _, _ = stats.binned_statistic(
        t, msd * 2 / ns, statistic='mean', bins=bin_edges)
    binned_msd_fast, _, _ = stats.binned_statistic(
        t, msd_fast * 2 / ns, statistic='mean', bins=bin_edges)
    binned_msd_slow, _, _ = stats.binned_statistic(
        t, msd_slow * 2 / ns, statistic='mean', bins=bin_edges)

    # Remove any NaN values from binned MSD data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid_all = ~np.isnan(binned_msd)
    binned_msd = binned_msd[valid_all]
    bin_centers_all = bin_centers[valid_all]

    valid_fast = ~np.isnan(binned_msd_fast)
    binned_msd_fast = binned_msd_fast[valid_fast]
    bin_centers_fast = bin_centers[valid_fast]

    valid_slow = ~np.isnan(binned_msd_slow)
    binned_msd_slow = binned_msd_slow[valid_slow]
    bin_centers_slow = bin_centers[valid_slow]

    # Plotting (Combined Analytical and Simulation)
    plt.figure(figsize=(9, 6))

    # Analytical Plot
    x = analytical_msd_df['t']
    y_columns = ['MSD_A', 'MSD_B', 'MSD_overall']
    y_label = ['$MSD_{fast}$', '$MSD_{slow}$', '$MSD_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, analytical_msd_df[column], label=label, color=color)
    plt.plot(x, analytical_msd_df['shortLim'], linestyle='--', color='black')
    plt.plot(x, analytical_msd_df['shortLim2'], linestyle='--', color='black')
    plt.plot(x, analytical_msd_df['MSDLongLim'], linestyle='--', color='black')

    # Simulation Scatter Plot (with binned means)
    plt.scatter(bin_centers_all[1:], binned_msd[1:],
                color="orange", marker='*')
    plt.scatter(bin_centers_fast[1:],
                binned_msd_fast[1:], color="red", marker='o')
    plt.scatter(bin_centers_slow[1:],
                binned_msd_slow[1:], color="blue", marker='^')

    # Plot Setup
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t, ns")
    # plt.ylabel("$MSD$", fontsize=20)
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$')
    # plt.grid(True)
    plt.xlim((0.001, 1000))
    # plt.legend(fontsize=14)
    plt.show()

# vectorised simulation plot


def plot_num(result):

    plt.figure(figsize=(9, 6))
    plt.plot(
        result["msd"]["t"][1:],
        (result["msd"]["status"][:, 1:] == 0).sum(axis=0),
        color="red",
        label="$N_{fast}$"
    )
    plt.plot(
        result["msd"]["t"][1:],
        (result["msd"]["status"][:, 1:] == 1).sum(axis=0),
        color="blue",
        label="$N_{slow}$"
    )
    plt.plot(
        result["msd"]["t"][1:],
        ((result["msd"]["status"][:, 1:] == 0) +
         (result["msd"]["status"][:, 1:] == 1)).sum(axis=0),
        color="orange",
        label="$N_{fast} + N_{slow}$"
    )

    # plt.legend(fontsize=14)
    plt.ylabel("$N$", fontsize=20)
    plt.xlabel("$t, ns$", fontsize=20)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim((0.001, 1000))
    # plt.grid(True)

# vectorised simulation plot


def plot_num_with_analytical_bin(result, analytical_popul_df, ylim_start=1, ylim_end=1000, num_bins=20):
    t = result["msd"]['t'][1:]

    # Get the number of free, trapped, and overall trajectories at each time step
    # / len(result["msd"]["status"])
    num_fast = np.sum(result["msd"]["status"][:, 1:] == 0, axis=0)
    # / len(result["msd"]["status"])
    num_slow = np.sum(result["msd"]["status"][:, 1:] == 1, axis=0)
    num_total = num_fast + num_slow

    # Logarithmic Binning for Simulation Data
    bin_edges = np.logspace(np.log10(min(t)), np.log10(max(t)), num_bins + 1)

    # Binned Mean Calculation with NaN Filtering
    binned_num_fast, _, _ = stats.binned_statistic(
        t, num_fast, statistic='mean', bins=bin_edges)
    binned_num_slow, _, _ = stats.binned_statistic(
        t, num_slow, statistic='mean', bins=bin_edges)
    binned_num_total, _, _ = stats.binned_statistic(
        t, num_total, statistic='mean', bins=bin_edges)

    # Remove any NaN values from binned data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    valid_fast = ~np.isnan(binned_num_fast)
    binned_num_fast = binned_num_fast[valid_fast]
    bin_centers_fast = bin_centers[valid_fast]

    valid_slow = ~np.isnan(binned_num_slow)
    binned_num_slow = binned_num_slow[valid_slow]
    bin_centers_slow = bin_centers[valid_slow]

    valid_total = ~np.isnan(binned_num_total)
    binned_num_total = binned_num_total[valid_total]
    bin_centers_total = bin_centers[valid_total]

    # Plotting (Combined Analytical and Simulation)
    plt.figure(figsize=(10, 6))

    # Analytical Plot
    x = analytical_popul_df['t']
    y_columns = ['num_a', 'num_b', 'num_total']
    y_label = ['$N_{fast}$', '$N_{slow}$', '$N_{overall}$']
    colors = ['red', 'blue', 'orange']
    markers = ['o', '^', '*']
    for column, label, color in zip(y_columns, y_label, colors):
        plt.plot(x, analytical_popul_df[column], label=label, color=color)
    plt.plot(x, analytical_popul_df['shortN1'], linestyle='--', color='black')
    plt.plot(x, analytical_popul_df['shortN2'], linestyle='--', color='black')
    plt.plot(x, analytical_popul_df['shortN12'], linestyle='--', color='black')
    plt.plot(x, analytical_popul_df['LongLimN1'],
             linestyle='--', color='black')
    plt.plot(x, analytical_popul_df['LongLimN2'],
             linestyle='--', color='black')
    plt.plot(x, analytical_popul_df['LongLimN12'],
             linestyle='--', color='black')

    # Simulation Scatter Plot (with binned means)
    plt.scatter(bin_centers_fast[1:], binned_num_fast[1:],
                color="red", label="fast QP", marker='o')
    plt.scatter(bin_centers_slow[1:], binned_num_slow[1:],
                color="blue", label="slow QP", marker='^')
    plt.scatter(bin_centers_total[1:], binned_num_total[1:],
                color="orange", label="all QP", marker='*')

    # Plot Setup
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("t, ns")
    plt.ylabel("$N$")
    # plt.grid(True)
    plt.xlim((0.001, 1000))
    plt.ylim((ylim_start, ylim_end))
    # plt.legend(fontsize=14)
    plt.show()

#######################################
def analyze_trajectories(trajectories, ns=10_000_000_000):
    max_time_steps = trajectories[0]['t'].size
    msd_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}
    population_data = {'A': np.zeros(max_time_steps), 'B': np.zeros(
        max_time_steps), 'overall': np.zeros(max_time_steps)}

    for traj in trajectories:
        initial_pos = np.array([traj['x'][0], traj['y'][0]])
        for t, (x, y, state) in enumerate(zip(traj['x'], traj['y'], traj['state'])):
            displacement = (x - initial_pos[0])**2 + (y - initial_pos[1])**2
            if state:  # Check if state is not empty
                msd_data[state][t] += displacement
                msd_data['overall'][t] += displacement
                population_data[state][t] += 1
                population_data['overall'][t] += 1

    time_steps = trajectories[0]['t']
    print(time_steps)
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'orange']

    for state, color in zip(['A', 'B', 'overall'], colors):
        msd_data[state] /= population_data[state]
        plt.plot(time_steps[2:], msd_data[state][2:]*2/ns,
                    label=f'MSD: {state}',color=color)  # /len(trajectories), , marker=marker, 
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    plt.xlim(min(time_steps), max(time_steps))
    plt.xlabel('$t, ns$') 
    plt.ylabel(
        '$\\langle \\mathrm{r}^2(t) \\rangle, \\mu m^2$') 
    # plt.title(f"Mean Square Displacement over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'orange']

    for state, color in zip(['A', 'B', 'overall'], colors):
        plt.plot(time_steps[1:], population_data[state][1:],
                 label=f'Population: {state}',  color=color)  # marker=marker,
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.001, 1000)
    plt.xlim(min(time_steps), max(time_steps))
    plt.xlabel('$t, ns$') 
    plt.ylabel('$N$')
    # plt.title(f"Population over Time, $\\alpha$={alpha}, $\\lambda_A$={lambda_A}, $\\lambda_B$={lambda_B}, $D_A$={tau_A}, $D_B$={tau_B}")
    # plt.legend()
    plt.show()
