#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:12:06 2021

@author: liang
"""
from utils import *

config_path = '/media/liang/Nas/PreTrainModel/simbert/chinese_simbert_L-4_H-312_A-12/bert_config.json' 
dict_path = '/media/liang/Nas/PreTrainModel/simbert/chinese_simbert_L-4_H-312_A-12/vocab.txt' 
modol_save_path = ''

# 建立分词器
tokenizer = get_tokenizer(dict_path)

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


from fea_utils import fea_map, fea_enum, fea_numeric, fea_text

def construct_string(data):
    
    numeric_string = ''.join(data[[fea_map[item] for item in fea_numeric]])
    enum_string = ''.join(data[[fea_map[item] for item in fea_enum]])
    
    mkg_string = ''.join(data[[fea_map[item] for item in fea_text]] )
    m_kg = textrank(mkg_string, withWeight=False, topK=20)
    
    keyword_string = ''.join(m_kg)
    
    return enum_string, numeric_string, keyword_string
    

def convert_to_vecs_batch(data, encoder, maxlen=64):
    """转换数据为id形式
    """
    resDF = pd.DataFrame()
    resDF['item_id'] = data['item_id']
    
    token_ids= []
    for index, item in data.iterrows():
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
    resDF['vecs'] = vecs
    return resDF


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
    
    
def cal_predict(fea_data, encoder):
    
    result = {}
    result['item_id'] = item_id
    item_otherlist = fea_data.keys()
    
        
def cal_whitening_predict(fea_data, encoder):
    
    
    result = {}
    result['item_id'] = item_id
    item_otherlist = fea_data.keys()
    vecs_list = []
    a_token_ids = tokenizer.encode(fea_data[item_id], maxlen=maxlen)[0]
    
    # 计算变换矩阵和偏置项
    kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])