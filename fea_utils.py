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
fea_map = {'币种-24h涨幅': 'DAY_PRICE_CHANGE_RATE',
             '币种-24h涨幅rank': 'DAY_PRICE_CHANGE_RATE_RANK',
             '币种-一周涨幅': 'WEEK_PRICE_CHANGE_RATE',
             '币种-一周涨幅rank': 'WEEK_PRICE_CHANGE_RATE_RANK',
             '币种-24h成交额': 'DAY_VOLUME_CHANGE_RATE',
             '币种-24h成交额rank': 'DAY_VOLUME_CHANGE_RATE_RANK',
             '币种-一周成交额': 'WEEK_VOLUME_CHANGE_RATE',
             '币种-一周成交额rank': 'WEEK_VOLUME_CHANGE_RATE_RANK',
             '币种-一周搜索热度rank': 'WEEK_CMC_SEARCH_HOT_RANK',
             '币种-24h点赞量': 'CURRENCY_LIKES',
             '币种-24h点赞量排名rank': 'CURRENCY_LIKES_RANKING',
             '币种-24h点赞量排名上升': 'CURRENCY_LIKES_RANKING_RISES_24_HOUR',
             '币种-24h点赞量排名上升rank': 'CURRENCY_LIKES_RANKING_RISES_24_HOUR_RANK',
             '币种-市值': 'CURRENCY_CAP',
             '币种-市值排名rank': 'CURRENCY_CAP_RANK',
             '币种-归属概念板块': 'CURRENCY_CATEGORY',
             '组合-24h收益率': 'INVEST_PROFIT_RATE_24_HOUR',
             '组合-24h收益率rank': 'INVEST_PROFIT_RATE_24_HOUR_RANK',
             '组合-一周收益率': 'INVEST_PROFIT_RATE_1_WEEK',
             '组合-一周收益率rank': 'INVEST_PROFIT_RATE_1_WEEK_RANK',
             '组合-年化收益率': 'INVEST_PROFIT_RATE_12_MONTH',
             '组合-年化收益率rank': 'INVEST_PROFIT_RATE_12_MONTH_RANK',
             '组合-关注量': 'AVERAGE_COLLECT',
             '组合-关注量rank': 'AVERAGE_COLLECT_RANK',
             '组合-使用量': 'INVEST_PROFIT_USED',
             '组合-使用量rank': 'INVEST_PROFIT_USED_RANK',
             '概念板块-24h涨幅': 'PLATE_CAP_CHANGE_RATE_1_DAY',
             '概念板块-24h涨幅rank': 'PLATE_CAP_CHANGE_RATE_1_DAY_RANK',
             '概念板块-一周涨幅': 'PLATE_CAP_CHANGE_RATE_1_WEEK',
             '概念板块-一周涨幅rank': 'PLATE_CAP_CHANGE_RATE_1_WEEK_RANK',
             '概念板块-收藏量': 'PLAT_COLLECT',
             '概念板块-收藏量rank': 'PLAT_COLLECT_RANK',
             '榜单类型': 'RANKLIST_TYPE',
             'item的推荐识别码': 'UUID'}

fea_enum = ['币种-24h涨幅rank',
                 '币种-一周涨幅rank',
                 '币种-24h成交额rank',
                 '币种-一周成交额rank',
                 '币种-一周搜索热度rank',
                 '币种-24h点赞量排名rank',
                 '币种-24h点赞量排名上升rank',
                 '币种-市值排名rank',
                 '组合-24h收益率rank',
                 '组合-一周收益率rank',
                 '组合-年化收益率rank',
                 '组合-关注量rank',
                 '组合-使用量rank',
                 '概念板块-24h涨幅rank',
                 '概念板块-一周涨幅rank',
                 '概念板块-收藏量rank']

fea_numeric = ['币种-24h成交额',
             '组合-24h收益率',
             'item的推荐识别码',
             '币种-24h点赞量',
             '榜单类型',
             '组合-关注量',
             '币种-24h涨幅',
             '概念板块-24h涨幅',
             '币种-一周涨幅',
             '币种-24h点赞量排名上升',
             '币种-一周成交额',
             '组合-使用量',
             '概念板块-一周涨幅',
             '币种-归属概念板块',
             '组合-一周收益率',
             '币种-市值',
             '组合-年化收益率',
             '概念板块-收藏量']

fea_text = ['详情描述', '评论内容']