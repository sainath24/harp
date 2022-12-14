import pandas as pd
import numpy as np
import re, os, math, copy

PRINT=False


def policy(model1, model2, w=1):
    better_model = None
    # 11, 12, 13, 14
    if model1[10] <= model2[10] and model1[11] <= model2[11] and model1[12] <= model2[12] and model1[13] <= model2[13]:
        better_model = model1
    elif model1[10] > model2[10] and model1[11] > model2[11] and model1[12] > model2[12] and model1[13] > model2[13]:
        better_model = model1
    else:
        UPPchange = w *abs(model1[10]-model2[10])/model1[10]
        MAEchange = abs(model1[11]-model2[11])/model2[11] 
        RMSEchange = abs(math.sqrt(model1[12])-math.sqrt(model2[12]))/math.sqrt(model2[12]) 
        MAPEchange = abs(model1[13]-model2[13])/model2[13]
        MAPEchangen = abs(model1[13]-model2[13])
        if PRINT:
            print(round(MAEchange,2), "<=", round(UPPchange,2), "and", round(RMSEchange,2), "<=", round(UPPchange,2), " and ",
                 round(MAPEchange, 2), "<=", round(UPPchange,2),)
        if (MAEchange <= UPPchange  and  RMSEchange <= UPPchange and MAPEchange <= UPPchange):
            better_model = model1
        else:
            better_model = model2
    return better_model
        


def get_better_model(model1, model2):
    # 0-UPP 1-MAE 2-MSE 3-MAPE
    # 0-APP    1-TD   2-RM   3-VA     4-BVE    5-UPP    6-MAE      7-MSE   8-MAPE
    better_model = None
    FOUND = False
    if PRINT:
        print("************************")
        print("model1", model1[10], model1)
        print("model2", model2[10], model2)
    better_model = policy(model1, model2, w=1)
    if PRINT:
        print("better", better_model)
        print("************************")
    return  better_model


def adjust_m1_m2(one, two):   
    if one[10] < two[10]:
        model1 = one
        model2 = two
    else:
        model1 = two
        model2 = one
    return (model1, model2)

def get_best_model(list_of_models):
    l_of_m = list_of_models.copy()
    model1 =l_of_m[l_of_m.UPP == l_of_m.UPP.min()]      
    l_of_m.drop(model1.index, axis=0,inplace=True)
    model1 = ((model1).values.tolist())[0]
    model2 =l_of_m[l_of_m.MAPE == l_of_m.MAPE.min()]   
    l_of_m.drop(model2.index, axis=0,inplace=True)
    model2 =((model2).values.tolist())[0]
    # if we have more than three models to choose from
    while l_of_m.shape[0] > 0:
        better_model = get_better_model(model1, model2)
        if model1 == better_model:
            # if the lower MAPE model is NOT better than model1, 
            # then pick the next lower MAPE model
            next_model = l_of_m[l_of_m.MAPE == l_of_m.MAPE.min()]
            l_of_m.drop(next_model.index, axis=0,inplace=True)
            next_model = ((next_model).values.tolist())[0]
            model1, model2 = adjust_m1_m2(better_model, next_model)
        else: 
            # if the lower MAPE model is better than model1, 
            # then pick the next lower UPP model
            next_model = l_of_m[l_of_m.UPP == l_of_m.UPP.min()]
            l_of_m.drop(next_model.index, axis=0,inplace=True)
            next_model = ((next_model).values.tolist())[0]
            model1, model2 = adjust_m1_m2(better_model, next_model)
    better_model = get_better_model(model1, model2)
    return better_model


def better_model_per_application(PMsMain, TD_LIST, SPLIT_LIST, RM_List):
    PMs = PMsMain.copy()
    PMs_RES = PMs.copy()
    PMs_RES["S1"] = ['-']*PMs.shape[0]
    PMs_RES["S2"] = ['-']*PMs.shape[0]
    PMs_RES["S3"] = ['-']*PMs.shape[0]
#     print(PMs_RES.head())
    PMs_level1 = []
    better_va_perTDRM = []
    for tdi in TD_LIST: 
        if tdi == "SD+n%FS":
            for spi in SPLIT_LIST: # TODO use all spi available in csv?
                for rm in RM_List:
                    l_of_m = PMs[(PMs['TDS'] == "SD") & (PMs['MOD'] == rm) & (PMs['n%FS'] == spi)] 
                    if l_of_m.shape[0] > 1:
                        one = (l_of_m.values.tolist())[0]
                        two = (l_of_m.values.tolist())[1]   
                        model1, model2 = adjust_m1_m2(one, two)
                        better_model = get_better_model(model1, model2)
                    else:
                            better_model = ((l_of_m).values.tolist())[0]
                            
                    better_va_perTDRM.append(better_model)  
                    idx = PMs.index[(PMs['APP']==better_model[0]) & (PMs['TDS']==better_model[1]) &
                                  (PMs['n%FS']==better_model[2]) & (PMs['FSO']==better_model[3]) &
                                   (PMs['MOD']==better_model[4]) & (PMs['VA']==better_model[5])].tolist()[0]
                    PMs_RES.at[idx,'S1']="O"
        else:
            spi = 50 if tdi == "FS" else 0
            for rm in RM_List:
                # VA_pRM_pTD
                l_of_m = PMs[(PMs['TDS'] == tdi) & (PMs['MOD'] == rm) & (PMs['n%FS'] == spi)] 
                if l_of_m.shape[0] > 1:
                    one = (l_of_m.values.tolist())[0]
                    two = (l_of_m.values.tolist())[1]   
                    model1, model2 = adjust_m1_m2(one, two)
                    better_model = get_better_model(model1, model2)
                else:
                    better_model = ((l_of_m).values.tolist())[0]
                better_va_perTDRM.append(better_model)  
                idx = PMs.index[(PMs['APP']==better_model[0]) & (PMs['TDS']==better_model[1]) &
                              (PMs['n%FS']==better_model[2]) & (PMs['FSO']==better_model[3]) &
                               (PMs['MOD']==better_model[4]) & (PMs['VA']==better_model[5])].tolist()[0]
                PMs_RES.at[idx,'S1']="O"

    PMs_level1_df = pd.DataFrame(better_va_perTDRM, 
                                 columns =  ['APP', 'TDS', 'n%FS', 'FSO', 'MOD', 'VA', 'MAE', 'MSE', 'MAPE', '#UP', 'UPP', 'MAEo', 'MSEo', 'MAPEo'])
    if PRINT:
        print("------------------------------------------------------", PMs_level1_df.shape)
        print(PMs_level1_df)
        print("------------------------------------------------------", PMs_level1_df.shape)
    PMs_level2 = []
    for rm in RM_List:
        # TD_pRM
        l_of_m = (PMs_level1_df[(PMs_level1_df['MOD'] == rm)]).copy()
        better_model = get_best_model(l_of_m)
        PMs_level2.append(better_model)
        idx = PMs.index[(PMs['APP']==better_model[0]) & (PMs['TDS']==better_model[1]) &
                                  (PMs['n%FS']==better_model[2]) & (PMs['FSO']==better_model[3]) &
                                   (PMs['MOD']==better_model[4]) & (PMs['VA']==better_model[5])].tolist()[0]
        PMs_RES.at[idx,'S2']="O"
        
    PMs_level2_df = pd.DataFrame(PMs_level2, 
                                 columns =  ['APP', 'TDS', 'n%FS', 'FSO', 'MOD', 'VA', 'MAE', 'MSE', 'MAPE', '#UP', 'UPP', 'MAEo', 'MSEo', 'MAPEo'])
    if PRINT:
        print("==========================================", PMs_level2_df.shape)
        print(PMs_level2_df)
        print("==========================================", PMs_level2_df.shape)
    PMs_level3 = []
    l_of_m = PMs_level2_df.copy()

    better_model = get_best_model(l_of_m)
    
    idx = PMs.index[(PMs['APP']==better_model[0]) & (PMs['TDS']==better_model[1]) &
                                  (PMs['n%FS']==better_model[2]) & (PMs['FSO']==better_model[3]) &
                                   (PMs['MOD']==better_model[4]) & (PMs['VA']==better_model[5])].tolist()[0]
    PMs_RES.at[idx,'S3']="O"
    
    PMs_level3.append(better_model)
    
    PMs_level3_df = pd.DataFrame(PMs_level3, 
                                 columns = ['APP', 'TDS', 'n%FS', 'FSO', 'MOD', 'VA', 'MAE', 'MSE', 'MAPE', '#UP', 'UPP', 'MAEo', 'MSEo', 'MAPEo'])
    return PMs_level3_df, PMs_RES


def model_selector(dataframe):    
    dataframe = dataframe[dataframe['TDS']!="FS"] #without full scale

    TD_LIST=["SD", "SD+n%FS"]
    SPLIT_LIST=[10, 20, 25, 30, 40, 50] #TODO include all splits
    RM_List=["NNR", "SLR", "DTR", "SVM"] #TODO change to model names

    pipeline_r, res_gr = better_model_per_application(dataframe, TD_LIST, SPLIT_LIST, RM_List)
    # res_gr.to_csv("GS_Selections.csv") #TODO change, do i even need to save this

    dataframe_BS = dataframe[dataframe['TDS']=="FS"] # with full scale
    baseline_r = get_best_model(dataframe_BS)
    BS_Model_df = pd.DataFrame([baseline_r, pipeline_r.values.tolist()[0]], 
                                    columns = dataframe.columns)

    model1, model2 = adjust_m1_m2(baseline_r, pipeline_r.values.tolist()[0])
    bS_Model_r = get_better_model(model1, model2)
    res = pd.DataFrame([bS_Model_r], columns = dataframe.columns)


