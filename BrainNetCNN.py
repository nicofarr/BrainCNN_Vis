
# coding: utf-8

# In[6]:

import matplotlib.pyplot as plt
plt.interactive(False)
import tensorflow as tf
import h5py
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, callbacks, regularizers, initializers
from E2E_conv import *
#from utils import *

import numpy as np
import os


# Loading the LSD data 

# In[7]:

behavdir = "/Users/nicolasfarrugia/Documents/recherche/git/Gold-MSI-LSD77/behav"

X = np.load(os.path.join(behavdir,"X_y_lsd77_static_tangent.npz"))['X']


# We keep only the Gold MSI data for y, then we normalize it 

# In[43]:

from sklearn.preprocessing import normalize

y = np.load(os.path.join(behavdir,"X_y_lsd77_static_tangent.npz"))['y']
labels = np.load(os.path.join(behavdir,"X_y_lsd77_static_tangent.npz"))['labels']


print(labels[3],",",labels[4])

y_sub=y[:,[3,4]]
y_norm = normalize(y_sub,axis=0)


# In[44]:

#import matplotlib.pyplot as plt
#%matplotlib inline 

#plt.subplot(2,2,1)
#plt.hist(y_sub[:,1])

#plt.subplot(2,2,2)
#plt.hist(y_sub[:,0])


#plt.subplot(2,2,3)
#plt.hist(y_norm[:,1])

#plt.subplot(2,2,4)
#plt.hist(y_norm[:,0])


# In[45]:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=0.33)


# Setting up the hyper parameters, and l2 regularizer
# 
# 
# 

# In[47]:

# Globals : Hyperparamaters
batch_size = 14
dropout = 0.5
momentum = 0.9
noise_weight = 0.125
lr = 0.01
decay = 0.0005

# Setting l2_norm regularizer
reg = regularizers.l2(decay)
kernel_init = initializers.he_uniform()


# In[48]:

# Model architecture 

n_feat = X.shape[1]

model = Sequential()
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,input_shape=(n_feat,n_feat,1),input_dtype='float32',data_format="channels_last"))
print("First layer output shape :"+str(model.output_shape))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(E2E_conv(2,32,(2,n_feat),kernel_regularizer=reg,data_format="channels_last"))
print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(64,(1,n_feat),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
model.add(Convolution2D(256,(n_feat,1),kernel_regularizer=reg,data_format="channels_last"))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))


model.add(Dense(128,kernel_regularizer=reg,kernel_initializer=kernel_init))
#print(model.output_shape)
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(30,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(LeakyReLU(alpha=0.33))
#print(model.output_shape)
model.add(Dropout(0.5))
model.add(Dense(2,kernel_regularizer=reg,kernel_initializer=kernel_init))
model.add(Flatten())
model.add(LeakyReLU(alpha=0.33))
model.summary()
#print(model.output_shape)


# ## Training The model
# - We use the euclidean distance as a cost function
# - The evaluation metric is the mean absolute error

# In[53]:

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


# In[54]:

opt = optimizers.SGD(momentum=momentum,nesterov=True,lr=lr)
model.compile(optimizer=opt,loss='mean_squared_error',metrics=['mae'])
csv_logger = callbacks.CSVLogger('BrainCNN_gmsi.log')

history=model.fit(X_train,y_train,epochs=1000,verbose=1,callbacks=[csv_logger])
model.save_weights("BrainCNN-gmsi.h5")

preds = model.predict(X_test)
##Epoch 1000/1000
#203/203 [==============================] - 3s - loss: 0.0617 - mean_absolute_error: 0.0127     
# In[ ]:

from scipy.stats import pearsonr

from sklearn.metrics import mean_absolute_error as mae


mae_1 = 100*mae(preds[:,0],y_test[:,0])
pears_1 = pearsonr(preds[:,0],y_test[:,0])
print("MAE for %s : %0.2f %%" % (labels[3],mae_1))
print("pearson R for %s : %0.2f, p = %0.2f" % (labels[3],pears_1[0],pears_1[1]))


mae_2 = 100*mae(preds[:,1],y_test[:,1])
pears_2 = pearsonr(preds[:,1],y_test[:,1])
print("MAE for %s : %0.2f %%" % (labels[4],mae_2))
print("pearson R for %s : %0.2f, p = %0.2f" % (labels[4],pears_2[0],pears_2[1]))

# 10 EPOCHS
#MAE for GoldMSI_Active_sum : 1.54 %
#pearson R for GoldMSI_Active_sum : 0.17, p = 0.08
#MAE for GoldMSI_Training_sum : 2.17 %
#pearson R for GoldMSI_Training_sum : -0.04, p = 0.72

## 100 EPOCHS 
#MAE for GoldMSI_Active_sum : 1.54 %
##pearson R for GoldMSI_Active_sum : 0.32, p = 0.00
#MAE for GoldMSI_Training_sum : 2.03 %
#pearson R for GoldMSI_Training_sum : 0.24, p = 0.02

#1000 EOOCH  
#MAE for GoldMSI_Active_sum : 1.32 %
#pearson R for GoldMSI_Active_sum : 0.52, p = 0.00
#MAE for GoldMSI_Training_sum : 1.57 %
#pearson R for GoldMSI_Training_sum : 0.69, p = 0.00


