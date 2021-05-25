#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:58:43 2021

@author: liang
"""

from pyhive import hive
 
conn = hive.Connection(host='HiveServer2 host', port=10000, username='hdfs', database='default')
cursor = conn.cursor()
cursor.execute('select * from demo_table limit 10')
for result in cursor.fetchall():
    print(result)