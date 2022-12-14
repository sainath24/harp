import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def write_csv(dataset, filename: str):
    dataset.to_csv(filename, header=True, index=False)


def get_file_size(setting, file_type):
    set_l = []
    mul=1
    rr_count=0
    if setting == "PE":
        mul=2
    
    if 'SRR8293759.1_1_40k.fastq.gz' in file_type:
        rr_count=10000
    elif 'SRR8293759.1_1_4L.fastq.gz' in file_type:
        rr_count=100000
    elif 'SRR8293759.1_1_4M.fastq.gz' in file_type:
        rr_count=1000000
    elif 'SRR8293759.1_1_4k.fastq.gz' in file_type:
        rr_count=1000
    elif 'SRR8293759.1_2_40k.fastq.gz' in file_type:
        rr_count=10000
    elif 'SRR8293759.1_2_4L.fastq.gz' in file_type:
        rr_count=100000
    elif 'SRR8293759.1_2_4M.fastq.gz' in file_type:
        rr_count=1000000
    elif 'SRR8293759.1_2_4k.fastq.gz' in file_type:
        rr_count=1000
    elif 'ERR875320.1_1_40k.fastq.gz' in file_type:
        rr_count=40000
    elif 'SRR10013451.1_1_4M.fastq.gz' in file_type:
        rr_count=400000
    elif 'ERR875320.1_1.fastq.gz' in file_type:
        rr_count=173641936
    elif 'SRR12514558.1_1_4M.fastq.gz' in file_type:
        rr_count=400000
    elif 'SRR10013451.1_1.fastq.gz' in file_type:
        rr_count=177013586
    elif 'SRR12514558.1_1.fastq.g' in file_type:
        rr_count=164846758
    else:
        set_l.append(file_type)

    return rr_count*mul


def remove_outliers(dataset):
    all_data_final = dataset
    # Removing Outliers:
    """
    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]
    all_data_final_noOut=all_data_final[(np.abs(temp) < 2).all(axis=1)]
    """
    scaler = StandardScaler()
    scaler.fit(all_data_final)
    temp = scaler.transform(all_data_final)
    # print(np.abs(stats.zscore(df)))
    all_data_final_noOut = all_data_final[(np.abs(temp) < 3).all(axis=1)]
    return all_data_final_noOut


def pca(dataset):
    """
    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]
    all_data_final_noOut=all_data_final[(np.abs(temp) < 2).all(axis=1)]
    """
    Dist_cols = ['sys_phy_cores_count',
        'sys_tot_cores_count', 'sys_cpufreq_mhz', 'sys_phy_mem_bytes',
        'sys_swap_mem_bytes', 'run_threads', 'run_seedMismatch', 'run_palindromeClipThreshold',
        'run_simpleClipThreshold', 'run_leading',
        'run_sliding_size', 'run_sliding_quality', 'run_minlen',
        'run_mi_length', 'run_mi_strict',
        'max_bytes_read_mb', 'max_bytes_written_mb',
        'max_read_bw_mbps', 'max_write_bw_mbps', 'data_rr_count']
    target_colum = 'total_exe_time_s'

    all_data_final_noOut = dataset
    X_Features_noOut = all_data_final_noOut[Dist_cols]
    x = X_Features_noOut.values
    X_Features_noOut_norm = StandardScaler().fit_transform(x)
    all_data_final_noOut_norm = pd.DataFrame(X_Features_noOut_norm, columns=Dist_cols)

    no_of_fetures = 12
    pca_n = PCA(n_components=no_of_fetures, random_state=2020)
    pca_n.fit(X_Features_noOut_norm)
    x_pca_n = pca_n.transform(X_Features_noOut_norm)


    df_pca = pd.DataFrame(x_pca_n)
    df_pca['walltime'] = all_data_final_noOut[target_colum].values

    return df_pca


# "training_data.csv"
def preprocess(dataset_list: list):
    assert type(dataset_list) == list, f"argument passed to trimmomatic data \
        preprocessor is not a list of dataset names, it is {type(dataset_list)}"
    data_l= dataset_list
    frames = []
    for fi in data_l:
        fi_open = "trimmomatic/"+fi
        df = pd.read_csv(fi_open)
        frames.append(df)

    data = pd.concat(frames)

    imp_columns=['run_config', 'sys_name', 'sys_processor', 'sys_phy_cores_count',
        'sys_tot_cores_count', 'sys_cpufreq_mhz', 'sys_phy_mem_bytes',
        'sys_swap_mem_bytes', 'run_settings', 'run_threads', 'run_input1',
        'run_input2', 'run_outputF_1P', 'run_outputF_1U', 'run_outputR_1P',
        'run_outputR_1U', 'run_seedMismatch', 'run_palindromeClipThreshold',
        'run_simpleClipThreshold', 'run_leading', 'run_trailing',
        'run_sliding_size', 'run_sliding_quality', 'run_minlen',
        'run_mi_length', 'run_mi_strict', 'total_exe_time_s',
        'total_wt_tau_time_ms', 'max_bytes_read_mb', 'max_bytes_written_mb',
        'max_read_bw_mbps', 'max_write_bw_mbps']

    target_colum = 'total_exe_time_s'


    rslt_df = data[(data['run_input1'].str.contains("_1.fastq.gz")) | (data['run_input1'].str.contains("_2.fastq.gz"))] #data['sys_cpufreq_mhz'] > 3000

    rslt_df1 = data[(data['sys_cpufreq_mhz'] >= 3000)] #data['sys_cpufreq_mhz'] > 3000
    rslt_df2 = data[(data['sys_cpufreq_mhz'] < 3000)] #data['sys_cpufreq_mhz'] > 3000

            
    data["data_rr_count"] = data.apply(lambda x: get_file_size(x["run_settings"], x["run_input1"]), axis = 1)

    rslt_df3 = data[(data['data_rr_count'] >= 1000000)] #data['sys_cpufreq_mhz'] > 3000

    Dist_cols = ['sys_phy_cores_count',
        'sys_tot_cores_count', 'sys_cpufreq_mhz', 'sys_phy_mem_bytes',
        'sys_swap_mem_bytes', 'run_threads', 'run_seedMismatch', 'run_palindromeClipThreshold',
        'run_simpleClipThreshold', 'run_leading',
        'run_sliding_size', 'run_sliding_quality', 'run_minlen',
        'run_mi_length', 'run_mi_strict',
        'max_bytes_read_mb', 'max_bytes_written_mb',
        'max_read_bw_mbps', 'max_write_bw_mbps', 'data_rr_count']
    target_colum = 'total_exe_time_s'


    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]

    all_data_final_noOut = remove_outliers(all_data_final)

    write_csv(all_data_final_noOut, 'trimmomatic_nopca.csv')

    df_pca = pca(all_data_final_noOut)
    
    return df_pca