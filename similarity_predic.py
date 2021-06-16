#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:12:06 2021

@author: liang
"""
from utils import *
from fea_utils import fea_map, fea_enum, fea_numeric, fea_text
import pandas as pd 
import os 
import itertools
from tqdm import tqdm
import copy
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

config_path = '/media/liang/Nas/PreTrainModel/retrive_genarate/simbert/chinese_simbert_L-4_H-312_A-12/bert_config.json' 
dict_path = '/media/liang/Nas/PreTrainModel/retrive_genarate/simbert/chinese_simbert_L-4_H-312_A-12/vocab.txt' 
modol_save_path = './best_model.weights'

# 建立分词器
tokenizer = get_tokenizer(dict_path)

model_type=['SimBERT-tiny', 'SimBERT-small'][0]
pooling = ['first-last-avg', 'last-avg', 'cls', 'pooler'][0]
dropout_rate = 0.3

# 建立模型
encoder = get_encoder(
    config_path,
    None,
    pooling=pooling,
    dropout_rate=dropout_rate
)
encoder.load_weights(modol_save_path)
    

def convert_to_vecs_one(data, encoder, maxlen=64):
    """转换数据为vec形式
    """
    vec_results = {}
    for id_item, value in data.items():
        token_ids, seg_ids = tokenizer.encode(value, maxlen=maxlen)
        vecs = encoder.predict([token_ids,seg_ids])
        vec_results[id_item] = vecs
    return vec_results


def construct_string1(data):
    
    numeric_string = ''.join(data[[fea_map[item] for item in fea_numeric]])
    enum_string = ''.join(data[[fea_map[item] for item in fea_enum]])
    
    mkg_string = ''.join(data[[fea_map[item] for item in fea_text]] )
    m_kg = textrank(mkg_string, withWeight=False, topK=20)
    
    keyword_string = ''.join(m_kg)
    
    return enum_string, numeric_string, keyword_string

   

def convert_to_vecs_batch(data,id_name, encoder, maxlen=64):
    """转换数据为id形式
    """
    vec_res = {}
    token_ids= []
    for index, item in tqdm(data.iterrows()):
        enum_string, numeric_string, keyword_string = construct_string(item)
        
        enum_token_id = tokenizer.encode(enum_string, maxlen=maxlen)[0]
        numeric_token_id = tokenizer.encode(numeric_string, maxlen=maxlen)[0][1:]
        keyword_token_id = tokenizer.encode(keyword_string, maxlen=maxlen)[0][1:]
        
        token_id = enum_token_id + [tokenizer.token_to_id('[SEP]')] + numeric_token_id +[tokenizer.token_to_id('[SEP]')] + keyword_token_id
        
        token_ids.append(token_id)
    token_ids = sequence_padding(token_ids)
    vecs = encoder.predict([token_ids,
                            np.zeros_like(token_ids)],
                            verbose=True)
    
    for id_, vec in zip(data[id_name], vecs):
        vec_res[id_] = vec
        
    return vec_res


def simcse_cal_func(a_vecs, b_vecs):
    a_vecs = l2_normalize(a_vecs)
    b_vecs = l2_normalize(b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    
    return sims



    
def whitening_cal_func(a_vecs, b_vecs, kernel, bias):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    
    return sims
        
def cal_whitening_predict(fea_data, encoder):
    
    result = {}
    result['item_id'] = item_id
    item_otherlist = fea_data.keys()
    vecs_list = []
    a_token_ids = tokenizer.encode(fea_data[item_id], maxlen=maxlen)[0]
    
    # 计算变换矩阵和偏置项
    kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
    
    
def cal_predict(fea_data, id_name):
    
    item_ids = fea_data[id_name].tolist()
    res_vecs = convert_to_vecs_batch(fea_data, id_name, encoder)
    
    sim_res = {}
    for item_a in  item_ids:
        
        sim_array = []
        tmp = copy.deepcopy(item_ids)
        tmp.remove(item_a)
        for item_b in tmp:
            sim = simcse_cal_func(res_vecs[item_a], res_vecs[item_b])
            sim_array.append((item_b, sim))
            
        result[item_a] = sim_array
            
    return result
                
    
    
    

if __name__ == '__main__':
    
    
    pred_df = pd.read_csv("/media/liang/Project2/推荐系统/git_code/deep_recommendation/data/item_fea.csv").drop_duplicates('物品ID')
    
    id_name = '物品ID'
    item_ids = pred_df[id_name].tolist()
    res_vecs = convert_to_vecs_batch(pred_df, id_name, encoder)
    
    itema_array = []
    itemb_array = []
    itema_vecs = []
    itemb_vecs = []
    
    # for item_a in  item_ids:
    #     sim_array = []
        
    #     tmp.remove(item_a)
    for item_a, item_b in itertools.product(item_ids, item_ids):
        if item_a != item_b:
            itema_array.append(item_a)
            itemb_array.append(item_b)
            
            itema_vecs.append(res_vecs[item_a].tolist())
            itemb_vecs.append(res_vecs[item_b].tolist())
            
    sim = simcse_cal_func(np.array(itema_vecs), np.array(itemb_vecs))
        # sim_array.append((item_b, sim))
    result = pd.DataFrame()
    result['item_id'] = itema_array
    result['similarity'] = [[item] for item in list(zip(itemb_array, sim))]
    
    result = result.groupby(['item_id']).sum()
    result.reset_index(drop=False,inplace=True)
    
    
    # result.reset_index()
            
    # return result
    
    # result = cal_predict(pred_df, '物品ID')    
    
    
    
    
    

        
    
    
    
    
    