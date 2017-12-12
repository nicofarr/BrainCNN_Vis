
# coding: utf-8

# In[1]:


from nilearn.connectome import ConnectivityMeasure
import numpy as np

# In[2]:


mask2mm = '../MNI152_T1_2mm_brain_mask.nii.gz'

# BASC parcellated data
# --

# In[75]:


#### Initialise all the data structures 
allscales = ['scale007','scale036','scale064','scale122','scale444']

tangent = []
pcorr  = []
preci = []
covar = []

# In[ ]:

for curscale in allscales:


    # In[3]:


    X = np.load('allsubjects_BASC_%s.npz' % curscale)['allts']
    subjids = np.load('allsubjects_excl_id.npz')['ids']


    # In[6]:


    if X.shape[0] != len(subjids):
        print('ERROR')


    # In[48]:


    i_excl=134

    # One subject (sub-010198) has a different length of time series... remove it 


    # There is also sub-010087 that has a length of 351 


    allsubjid = np.concatenate([subjids[:i_excl],subjids[(i_excl+1):149],subjids[150:]])

    # So in total we now have 158 subjects

    # In[59]:


    temp = X[:i_excl]
    temp = np.stack(temp)

    temp2 = X[i_excl+1:149]
    temp2 = np.stack(temp2)

    temp3 = X[150:]
    temp3 = np.stack(temp3)

    allts_final = np.stack(np.concatenate([temp,temp2,temp3],axis = 0))

    print("Calculating connectivity at %s" % curscale)
    conn = ConnectivityMeasure(kind='tangent')

    tangent = conn.fit_transform(allts_final)

    conn = ConnectivityMeasure(kind='partial correlation')

    pcorr = conn.fit_transform(allts_final)

    conn = ConnectivityMeasure(kind='precision')

    preci = conn.fit_transform(allts_final)

    conn = ConnectivityMeasure(kind='covariance')

    covar = conn.fit_transform(allts_final)
    np.savez_compressed("fconn_158subjects_%s_regions.npz" % curscale,
                        tangent=tangent,pcorr=pcorr,preci=preci,covar=covar,subjectids=allsubjid)
  