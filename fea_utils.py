#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:01:24 2021

@author: liang
"""

# import pandas as pd
# df = pd.read_csv('./data/item_fea_字典表.csv')
# fea_name = {}
# for key, value in zip(df['字段名称'], df['英文标识']):
#     fea_name[key] = value
fea_map = {'物品ID': 'item_id',
             '物品类别': 'item_type',
             '24 小时趋势': '24Htrend',
             '24 小时内上涨下跌比例': '24Htrendnum',
             '24 小时内上涨下跌排名': '24Htrendindex',
             '一周趋势': '1Wtrend',
             '一周内上涨下跌比例': '1Wtrendnum',
             '一周内上涨下跌排名': '1Wtrendindex',
             '市值': 'marketcap',
             '市值排名': 'marketcapindex',
             '24 小时内交易量': '24Hvolume',
             '一周内交易量': '1Wvolume',
             '24 小时内搜索排名': '24Hsearchindex',
             '24 小时内搜索排名变化': '24Hsearchchange',
             '一周内搜索排名': '1Wsearchindex',
             '一周内搜索排名变化': '1Wsearchchange',
             '归属的概念板块': 'category',
             '关注量': 'follownum',
             '关注量排行': 'followindex',
             '关注量排行变化': 'followindexchange',
             '三月内趋势': '3Mtrend',
             '三月内上涨下跌比例': '3Mtrendnum',
             '三月内上涨下跌排名': '3Mtrendindex',
             '成分币种数量': 'compositionnum',
             '详情描述': 'description',
             '评论内容': 'comment'}

fea_numeric = [  
                 '24 小时内上涨下跌比例',
                 '24 小时内上涨下跌排名',
                 '一周内上涨下跌比例',
                 '一周内上涨下跌排名',
                 '市值',
                 '市值排名',
                 '24 小时内交易量',
                 '一周内交易量',
                 '24 小时内搜索排名',
                 '24 小时内搜索排名变化',
                 '一周内搜索排名',
                 '一周内搜索排名变化',
                 '关注量',
                 '关注量排行',
                 '关注量排行变化',
                 '三月内上涨下跌比例',
                 '三月内上涨下跌排名',
                 '成分币种数量'
                 ]

fea_enum = [  
                 '物品类别',
                 '24 小时趋势',
                 '一周趋势',
                 '一周内上涨下跌比例',
                 '一周内上涨下跌排名',
                 '24 小时内搜索排名变化',
                 '一周内搜索排名',
                 '一周内搜索排名变化',
                 '归属的概念板块',
                 '关注量排行变化',
                 '三月内趋势'
                 ]

fea_text = ['详情描述', '评论内容']