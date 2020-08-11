# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 19:17:25 2019

@author: gzs13133
"""

import os
os.chdir('D://data//support//bilstm')

from model import biLSTMwithCRF

model = biLSTMwithCRF(ckpt_dir='checkpoint2/')
#model.load_data_one()
model.load_data()
model.build()
model.train()

model.load_model()
model.predict('小姐姐我爱你么么。')




