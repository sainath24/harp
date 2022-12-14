import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def write_csv(dataset, filename):
    dataset.to_csv(filename, header = True, index = False)
# do same but attach it to the dataframe
def get_file_size(file_type):
    rr_count=0
    if file_type == "1h":
        rr_count=100
    elif file_type == "2h":
        rr_count=200
    elif file_type == "4h":
        rr_count=400
    elif file_type == "1k":
        rr_count=1000
    elif file_type == "2k":
        rr_count=2000
    elif file_type == "5k":
        rr_count=5000
    elif file_type == "10k":
        rr_count=10000
    # else:
    #     print(file_type)
    return rr_count

def get_model_code(model_name):
    rr_count=0
#     vgg16' 'resnet' 'inception'
    if model_name == "vgg16":
        rr_count=1
    elif model_name == "resnet":
        rr_count=2
    elif model_name == "inception":
        rr_count=3
    return rr_count

def remove_outliers(dataset):
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
    all_data_final_noOut = all_data_final[(np.abs(temp) >=0 ).all(axis=1)]

    return all_data_final_noOut

def pca(dataset):
    """
    all_cols = Dist_cols+[target_colum]
    X_Features = data[Dist_cols]
    Y_value = data[[target_colum]]
    all_data_final=data[all_cols]
    all_data_final_noOut=all_data_final[(np.abs(temp) < 2).all(axis=1)]
    """
    Dist_cols = ['GPU', 'sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_ntasks_per_core', 'sys_cpufreq_mhz',
       'sys_phy_mem_bytes', 'sys_swap_mem_bytes', 'data_rr_count',
       'model_encoding', 'batch_size', 'epochs', 'learning_rate']
    targrt_column_list= ['epochs_times_one', 'epochs_times_avg', 'fit_time']

    all_data_final_noOut = dataset
    X_Features_noOut = all_data_final_noOut[Dist_cols]
    x = X_Features_noOut.values
    X_Features_noOut_norm = StandardScaler().fit_transform(x)
    all_data_final_noOut_norm = pd.DataFrame(X_Features_noOut_norm, columns=Dist_cols)

    no_of_fetures = min(X_Features_noOut_norm.shape[0], 9) #TODO: HAD TO INTRODUCE MIN HERE) BUT MODEL TRAINER EXPECTS 9
    pca_n = PCA(n_components=no_of_fetures, random_state=2020)
    pca_n.fit(X_Features_noOut_norm)
    x_pca_n = pca_n.transform(X_Features_noOut_norm)

    df_pca = pd.DataFrame(x_pca_n)
    df_pca['epochs_times_one'] = all_data_final_noOut[targrt_column_list[0]].values
    df_pca['epochs_times_avg'] = all_data_final_noOut[targrt_column_list[1]].values
    df_pca['fit_time'] = all_data_final_noOut[targrt_column_list[2]].values

    return df_pca

def preprocess(dataset_list: list):
    # "training_data.csv"
    data_l= dataset_list

    frames = []
    for fi in data_l:
        fi_open = fi
        df = pd.read_csv(fi_open)
        frames.append(df)

    data = pd.concat(frames)

    imp_columns=['GPU', 'sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_ntasks_per_core', 'sys_cpufreq_mhz',
       'sys_phy_mem_bytes', 'sys_swap_mem_bytes', 'file_type', 'CPU_GPU',
       'model_name', 'batch_size', 'epochs', 'learning_rate', 'FP16',
       'epochs_times', 'epochs_times_one', 'epochs_times_avg', 'test_accuracy',
       'fit_time']

    target_colum = 'total_exe_time_s'

    data['fit_time_F'] = data['fit_time']
    try:
        data["data_rr_count"] = data.apply(lambda x: get_file_size(x["file_type"]), axis = 1)
    except:
        data["data_rr_count"] = data.apply(lambda x: get_file_size(x["model_name"]), axis = 1)


    data["model_encoding"] = data.apply(lambda x: get_model_code(x["model_name"]), axis = 1)

    Dist_cols = ['GPU', 'sys_phy_cores_count',
       'sys_tot_cores_count', 'sys_ntasks_per_core', 'sys_cpufreq_mhz',
       'sys_phy_mem_bytes', 'sys_swap_mem_bytes', 'data_rr_count',
       'model_encoding', 'batch_size', 'epochs', 'learning_rate']
    targrt_column_list= ['epochs_times_one', 'epochs_times_avg', 'fit_time']
    # target_colum = 'epochs_times_avg'

    all_cols = Dist_cols+targrt_column_list
    X_Features = data[Dist_cols]
    Y_value = data[targrt_column_list]
    all_data_final=data[all_cols]

    # all_data_final=all_data_final.loc[all_data_final['model_encoding'] == 1] #TODO: THIS LEADS TO EMPTY DATA FRAME IF MODEL ENCODING NOT PRESENT

    all_data_final_noOut = remove_outliers(all_data_final)

    df_pca = pca(all_data_final_noOut)

    return df_pca

