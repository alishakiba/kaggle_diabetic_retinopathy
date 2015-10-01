
# coding: utf-8

# In[1]:

import sys
sys.path.append('../')


import cPickle as pickle
import re
import glob
import os

import time

import theano
import theano.tensor as T
import numpy as np
import pandas as p
import lasagne as nn

from PIL import Image

from utils import hms, architecture_string, get_img_ids_from_iter


# In[2]:

get_ipython().magic(u'pylab inline')
rcParams['figure.figsize'] = 16, 6
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


# In[3]:

dump_path = '../dumps/2015_07_17_123003_PARAMSDUMP.pkl'


# In[4]:

model_data = pickle.load(open(dump_path, 'r'))


# In[5]:

from models import basic_model as model


# In[6]:

LEARNING_RATE_SCHEDULE = model.LEARNING_RATE_SCHEDULE
prefix_train = model.prefix_train if hasattr(model, 'prefix_train') else     '/run/shm/train_ds2_crop/'
prefix_test = model.prefix_test if hasattr(model, 'prefix_test') else     '/run/shm/test_ds2_crop/'
SEED = model.SEED if hasattr(model, 'SEED') else 11111

id_train, y_train = model.id_train, model.y_train
id_valid, y_valid = model.id_valid, model.y_valid
id_train_oversample = model.id_train_oversample,
labels_train_oversample = model.labels_train_oversample

sample_coefs = model.sample_coefs if hasattr(model, 'sample_coefs')     else [0, 7, 3, 22, 25]

l_out, l_ins = model.build_model()


# In[7]:

# params = nn.layers.get_all_param_values(l_out)
# for p, v in zip(params, model_data):
#     p = v
# l_out.params = model_data


# In[8]:

nn.layers.set_all_param_values(l_out, model_data)


# In[9]:

chunk_size = 64
batch_size = 64


# In[10]:

output = nn.layers.get_output(l_out, deterministic=True)
input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]


# In[11]:

import pandas as pd
OriginalLabels = pd.read_csv(r'../data/trainLabels.csv', sep=',')


# In[12]:

import glob
temp1 = np.zeros((chunk_size, 3, 512, 512), dtype='float64')
fileList = sorted(glob.glob(r'/home/ali/Desktop/kaggle_diabetic_retinopathy-master/data/64/*.tiff'))
labels64 = []
for i, f in enumerate(fileList):
    temp1[i,:,:,:] = np.array(Image.open(f)).T / 255.0
    fname = f.split('/')[-1].split('.')[0]
    lbl = OriginalLabels.loc[OriginalLabels['image'] == fname]['level'].values.item(0)
#     print lbl
    labels64.append(lbl)
xs_shared[0].set_value(temp1)
print temp1.shape


# In[13]:

temp2 = np.ones((chunk_size, 2), dtype='float64') * 512
xs_shared[1].set_value(temp2)
temp2.shape


# In[15]:

idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[idx * batch_size:(idx + 1) * batch_size]


# In[ ]:

compute_output = theano.function(
    [idx],
    output,
    givens=givens,
    on_unused_input='ignore'
)


                import pandas as pd
OriginalLabels = pd.read_csv(r'../data/trainLabels.csv', sep=',')
                
                import glob
temp1 = np.zeros((chunk_size, 3, 512, 512), dtype='float64')
fileList = glob.glob(r'/home/ali/Desktop/kaggle_diabetic_retinopathy-master/data/64/*.tiff')
labels64 = []
for i, f in enumerate(fileList):
    temp1[i,:,:,:] = np.array(Image.open(f)).T / 255.0
    fname = f.split('/')[-1].split('.')[0]
    lbl = OriginalLabels.loc[OriginalLabels['image'] == fname]['level'].values.item(0)
#     print lbl
    labels64.append(lbl)
xs_shared[0].set_value(temp1)
print temp1.shape
                
                temp2 = np.ones((chunk_size, 2), dtype='float64') * 512
xs_shared[1].set_value(temp2)
temp2.shape
                
# In[ ]:

l_out.params


# In[ ]:

get_ipython().magic(u'time predictions = compute_output(0)')


# In[ ]:

results = []
for i, p in enumerate(predictions):
    resul = dict()
    resul['fileName'] = fileList[i].split('/')[-1].split('.')[0]
    resul['original'] = labels64[i]
    resul['pred'] = p
    results.append(resul)


# In[ ]:

for i in results:
    print i


# In[ ]:

len(model_data)


# In[ ]:




# In[ ]:




# In[ ]:



