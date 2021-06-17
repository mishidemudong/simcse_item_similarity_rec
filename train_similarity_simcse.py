#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:35:34 2021

@author: liang
"""

from utils import *
import sys
import os
import tensorflow as tf
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
import pandas as pd
import jieba
from jieba.analyse import textrank
jieba.initialize()
import threading
import multiprocessing
import copy
from multiprocessing import Pool

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

###set gpu memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# 基本参数
model_type=['SimBERT-tiny', 'SimBERT-small'][0]
pooling = ['first-last-avg', 'last-avg', 'cls', 'pooler'][0]
task_name = ['ATEC', 'BQ', 'LCQMC'][1]
dropout_rate = 0.1
epoch = 10
maxlen = 64

# bert配置
model_name = {
    'SimBERT-tiny': 'chinese_simbert_L-4_H-312_A-12',
    'SimBERT-small': 'chinese_simbert_L-6_H-384_A-12'
}[model_type]

config_path = '/media/liang/Nas/PreTrainModel/retrive_genarate/simbert/chinese_simbert_L-4_H-312_A-12/bert_config.json' 
checkpoint_path = '/media/liang/Nas/PreTrainModel/retrive_genarate/simbert/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt' 
dict_path = '/media/liang/Nas/PreTrainModel/retrive_genarate/simbert/chinese_simbert_L-4_H-312_A-12/vocab.txt' 

# 建立分词器
tokenizer = get_tokenizer(dict_path)

# 建立模型
if model_type == 'RoFormer':
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='roformer',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
elif 'NEZHA' in model_type:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        model='nezha',
        pooling=pooling,
        dropout_rate=dropout_rate
    )
else:
    encoder = get_encoder(
        config_path,
        checkpoint_path,
        pooling=pooling,
        dropout_rate=dropout_rate
    )



print_step = 0
class data_generator(DataGenerator):
    """训练语料生成器
    """
    def __iter__(self, random=False):
        global print_step
        batch_token_ids = []
        for is_end, token_ids in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_token_ids.append(token_ids)
            
            if print_step % 500 ==0:
                print()
                print(print_step)
                print("token_ids:", tokenizer.ids_to_tokens(token_ids))
                print("token_decode:", tokenizer.decode(token_ids))
            print_step += 1
            
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = np.zeros_like(batch_token_ids)
                batch_labels = np.zeros_like(batch_token_ids[:, :1])
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids = []


def simcse_loss(y_true, y_pred):
    """用于SimCSE训练的loss
    """
    # 构造标签
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = K.equal(idxs_1, idxs_2)
    y_true = K.cast(y_true, K.floatx())
    # 计算相似度
    y_pred = K.l2_normalize(y_pred, axis=1)
    similarities = K.dot(y_pred, K.transpose(y_pred))
    similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = K.categorical_crossentropy(y_true, similarities, from_logits=True)
    return K.mean(loss)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_model.weights')
        

if __name__ == '__main__':
    # 加载数据集
    trained_df = pd.read_csv("/media/liang/Project2/推荐系统/git_code/deep_recommendation/data/item_fea.csv").sample(10000)
    
        
    # 语料id化
    # train_token_ids= []
    # for index, item in tqdm(trained_df.iterrows()):
    #     enum_string, numeric_string, keyword_string = construct_string(item)
        
    #     enum_token_id = tokenizer.encode(enum_string, maxlen=maxlen)[0]
    #     numeric_token_id = tokenizer.encode(numeric_string, maxlen=maxlen)[0][1:]
    #     keyword_token_id = tokenizer.encode(keyword_string, maxlen=maxlen)[0][1:]
        
    #     token_id = enum_token_id + [tokenizer.token_to_id('[SEP]')] + numeric_token_id +[tokenizer.token_to_id('[SEP]')] + keyword_token_id
        
    #     train_token_ids.append(token_id)
    # train_token_ids = sequence_padding(train_token_ids)
    
    
    
    train_token_ids= []
    def MainRange(trained_df):     #提供列表index起始位置参数
        part_token_ids= []
        for index, item in tqdm(trained_df.iterrows()):
            enum_string, numeric_string, keyword_string = construct_string(item)
            
            enum_token_id = tokenizer.encode(enum_string, maxlen=maxlen)[0]
            numeric_token_id = tokenizer.encode(numeric_string, maxlen=maxlen)[0][1:]
            keyword_token_id = tokenizer.encode(keyword_string, maxlen=maxlen)[0][1:]
            
            token_id = enum_token_id + [tokenizer.token_to_id('[SEP]')] + numeric_token_id +[tokenizer.token_to_id('[SEP]')] + keyword_token_id
            
            part_token_ids.append(token_id)
        return part_token_ids
            
    df_parts=np.array_split(trained_df,20)
    print(len(df_parts),type(df_parts[0]))

    with Pool(processes=8) as pool:
        result_parts = pool.map(MainRange,df_parts)
        # pool.map(MainRange,df_parts)
    for item in result_parts:
        train_token_ids.extend(item)
    

    

    
    
    train_token_ids = sequence_padding(train_token_ids)
    
    # SimCSE训练
    evaluator = Evaluator()
    encoder.summary()
    encoder.compile(loss=simcse_loss, optimizer=Adam(1e-5))
    train_generator = data_generator(train_token_ids, 64)
    
    encoder.load_weights('./best_model.weights')
    encoder.fit(
        train_generator.forfit(), 
        # steps_per_epoch=len(train_generator), 
        steps_per_epoch=1000, 
        epochs=epoch,
        callbacks=[evaluator]
    )
    
    encoder.save_weights('./best_model.weights')
    
    
    
    ####evaluate
    # # 语料向量化
    # all_vecs = []
    # for a_token_ids, b_token_ids in all_token_ids:
    #     a_vecs = encoder.predict([a_token_ids,
    #                               np.zeros_like(a_token_ids)],
    #                              verbose=True)
    #     b_vecs = encoder.predict([b_token_ids,
    #                               np.zeros_like(b_token_ids)],
    #                              verbose=True)
    #     all_vecs.append((a_vecs, b_vecs))
    
    # 标准化，相似度，相关系数
    # all_corrcoefs = []
    # for (a_vecs, b_vecs), labels in zip(all_vecs, all_labels):
    #     print(a_vecs)
    #     a_vecs = l2_normalize(a_vecs)
    #     b_vecs = l2_normalize(b_vecs)
    #     sims = (a_vecs * b_vecs).sum(axis=1)
    #     corrcoef = compute_corrcoef(labels, sims)
    #     all_corrcoefs.append(corrcoef)
    
    # all_corrcoefs.extend([
    #     np.average(all_corrcoefs),
    #     np.average(all_corrcoefs, weights=all_weights)
    # ])
    
    # for name, corrcoef in zip(all_names + ['avg', 'w-avg'], all_corrcoefs):
    #     print('%s: %s' % (name, corrcoef))


    
    
        
        

    
    
    