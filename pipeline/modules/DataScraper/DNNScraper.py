import pandas as pd
import re, os, json
import psutil
import platform
from datetime import datetime


def scrape(data):
    df = pd.DataFrame(columns = [])
    meta_data = {}
    """
    sys_name,sys_processor,
    sys_phy_cores_count,sys_tot_cores_count,sys_cpufreq_mhz,
    sys_phy_mem_bytes,sys_swap_mem_bytes,

    """
    tmp_dir = os.getcwd()
    cur_dir = os.path.basename(os.path.normpath(tmp_dir))
    uname = platform.uname()
    meta_data['sys_name']=uname.system
    meta_data['sys_processor']=uname.processor
    cpufreq = psutil.cpu_freq()
    meta_data['sys_phy_cores_count']=psutil.cpu_count(logical=False)
    meta_data['sys_tot_cores_count']=psutil.cpu_count(logical=True)
    tmp = [cpufreq.max, cpufreq.min, cpufreq.current]
    meta_data['sys_cpufreq_mhz']=max(tmp)
    svmem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    meta_data['sys_phy_mem_bytes']=svmem.total
    meta_data['sys_swap_mem_bytes']=swap.total


    '''
    #TODO open this
    fob = {}
    with open('codar.cheetah.run-params.dnn.json') as f:
        fob = json.load(f)


    """
    file_type,CPU_GPU,model_name,batch_size,epochs,learning_rate,FP16
    """
    for exe in fob.keys():
        for setting in fob[exe].keys():
            if setting == "file_type":
                meta_data["file_type"]=fob[exe][setting]
            elif setting == "CPU_GPU":
                meta_data["CPU_GPU"]=fob[exe][setting]
            elif setting == "batch_size":
                meta_data["batch_size"]=int(fob[exe][setting])
            elif setting == "epochs":
                meta_data["epochs"]=int(fob[exe][setting])
            elif setting == "learning_rate":
                meta_data["learning_rate"]=float(fob[exe][setting])
            elif setting == "model_name":
                meta_data["model_name"]=fob[exe][setting]
    meta_data['FP16']=0 


    out_lines = []
    #TODO: open output
    with open('codar.workflow.stdout.dnn') as f:
        out_lines = f.readlines()


    """
    epochs_times,epochs_times_one,epochs_times_avg,test_accuracy,fit_time
    'MODEL EPOCHS TIME ON TEST: 135.78\n',
    'MODEL EPOCH ONE ON TEST: 135.78\n',
    'MODEL EPOCH AVG ON TEST: 135.78\n',
    'MODEL ACCURACY ON TEST: 0.31\n',
    'MODEL FIT ON TEST: 135.92\n'
    """

    for line in out_lines:
        line = line.strip()
        if "MODEL EPOCHS TIME ON TEST:" in line:
            line = re.sub(r"MODEL EPOCHS TIME ON TEST: ", "", line)
            meta_data['epochs_times']=float(line)
            print()
        elif "MODEL EPOCH ONE ON TEST:" in line:
            line = re.sub(r"MODEL EPOCH ONE ON TEST: ", "", line)
            meta_data['epochs_times_one']=float(line)
        elif "MODEL EPOCH AVG ON TEST:" in line:
            line = re.sub(r"MODEL EPOCH AVG ON TEST: ", "", line)
            meta_data['epochs_times_avg']=float(line)
        elif "MODEL ACCURACY ON TEST:" in line:
            line = re.sub(r"MODEL ACCURACY ON TEST: ", "", line)
            meta_data['test_accuracy']=float(line)
        elif "MODEL FIT ON TEST:" in line:
            line = re.sub(r"MODEL FIT ON TEST: ", "", line)
            meta_data['fit_time']=float(line)

    '''
    # meta_data["file_type"]=fob[exe][setting]
    meta_data["CPU_GPU"] = data["CPU_GPU"]
    meta_data["batch_size"] = data["batch_size"]
    meta_data["epochs"] = data["epochs"]
    meta_data["model_name"] = data["model_name"]
    meta_data["learning_rate"] = data["learning_rate"]

    meta_data['epochs_times_one'] = data["max_time_per_epoch"]
    meta_data['epochs_times_avg'] = data["avg_time_per_epoch"]
    meta_data['fit_time'] = data["total_training_time"]

    #TODO: FIND WHERE TO GET THESE
    meta_data['GPU'] = 1
    meta_data['sys_ntasks_per_core'] = 1
    meta_data['FP16']=0 
    meta_data["file_type"]="default"
    meta_data['test_accuracy'] = 1
    meta_data['epochs_times'] = data["max_time_per_epoch"]


    df = pd.DataFrame([meta_data])
    return df


# sys_name,sys_processor,sys_phy_cores_count,sys_tot_cores_count,sys_cpufreq_mhz,sys_phy_mem_bytes,sys_swap_mem_bytes,
# file_type,CPU_GPU,model_name,batch_size,epochs,learning_rate,FP16,epochs_times,epochs_times_one,epochs_times_avg,test_accuracy,fit_time
