import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def write_csv(dataset, filename: str):
    dataset.to_csv(filename, header = True, index = False)

def remove_outliers(dataset):
    assert dataset != None, "None passed as dataset to remove_outliers in GrayScottPreprocessor"
    # Removing Outliers:
    """
    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]
    all_data_final_noOut=all_data_final[(np.abs(temp) < 2).all(axis=1)]
    """
    all_data_final = dataset
    scaler = StandardScaler()
    scaler.fit(all_data_final)
    temp = scaler.transform(all_data_final)
    # print(np.abs(stats.zscore(df)))
    all_data_final_noOut = all_data_final[(np.abs(temp) < 3).all(axis=1)]

    return all_data_final_noOut

def pca(dataset):
    assert dataset != None, "None passed as dataset to pca in GrayScottPreprocessor"
    """
    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]
    all_data_final_noOut=all_data_final[(np.abs(temp) < 2).all(axis=1)]
    """
    Dist_cols = ['sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_phy_mem_bytes',
       'exe_L', 'exe_Du', 'exe_Dv', 'exe_F', 'exe_k',
       'exe_dt', 'exe_steps', 'exe_noise', 'run_nprocs', 
       'max_bytes_written_mb', 'max_read_bw_mbps',
       'max_write_bw_mbps']
    target_colum = 'total_exe_time_ms'

    all_data_final_noOut = dataset
    X_Features_noOut = all_data_final_noOut[Dist_cols]
    x = X_Features_noOut.values
    X_Features_noOut_norm = StandardScaler().fit_transform(x)
    all_data_final_noOut_norm = pd.DataFrame(X_Features_noOut_norm, columns=Dist_cols)

    no_of_fetures = 13
    pca_n = PCA(n_components=no_of_fetures, random_state=2020)
    pca_n.fit(X_Features_noOut_norm)
    x_pca_n = pca_n.transform(X_Features_noOut_norm)

    df_pca = pd.DataFrame(x_pca_n)
    df_pca['walltime'] = all_data_final_noOut[target_colum].values

    return df_pca

def preprocess(dataset_list: list):
    data_l= dataset_list

    frames = []
    for fi in data_l:
        fi_open = "GrayScott/"+fi
        df = pd.read_csv(fi_open)
        frames.append(df)

    data = pd.concat(frames)

    imp_columns= Dist_cols = ['sys_name', 'sys_processor', 'sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_cpufreq_mhz', 'sys_phy_mem_bytes',
       'sys_swap_mem_bytes', 'exe_L', 'exe_Du', 'exe_Dv', 'exe_F', 'exe_k',
       'exe_dt', 'exe_plotgap', 'exe_steps', 'exe_noise', 'exe_output',
       'exe_checkpoint', 'exe_checkpoint_freq', 'exe_checkpoint_output',
       'exe_adios_config', 'exe_adios_span', 'exe_adios_memory_selection',
       'exe_mesh_type', 'run_nprocs', 'total_exe_time_ms',
       'max_bytes_read_mb', 'max_bytes_written_mb', 'max_read_bw_mbps',
       'max_write_bw_mbps']

    target_colum = 'total_exe_time_ms'

    rslt_df = data[(data['exe_L'] > 100) & (data['exe_steps'] == 1000)] #data['sys_cpufreq_mhz'] > 3000 (data['exe_L'] >=64) &

    Small_scale = data[(data['exe_L'] > 100) & (data['exe_steps'] >= 1000)]
    TestL = data[(data['exe_L'] <= 100) | ((data['exe_L'] > 100) & (data['exe_steps'] < 1000))]

    Dist_cols = ['sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_phy_mem_bytes',
       'exe_L', 'exe_Du', 'exe_Dv', 'exe_F', 'exe_k',
       'exe_dt', 'exe_steps', 'exe_noise', 'run_nprocs', 
       'max_bytes_written_mb', 'max_read_bw_mbps',
       'max_write_bw_mbps']
    target_colum = 'total_exe_time_ms'

    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]

    all_data_final_noOut = remove_outliers(all_data_final)

    df_pca = pca(all_data_final_noOut)

    return df_pca



