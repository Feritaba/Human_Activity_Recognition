#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import sklearn.metrics as mt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

#from lightgbm import LGBMClassifier
#from xgboost import XGBClassifier
#from catboost import CatBoostClassifier


# In[2]:


df = pd.read_csv("KU-HAR_v1.0_raw_samples.csv")
dff = df.values
signals = dff[:, 0: 1800]                         #These are the time-domian subsamples (signals) 
signals = np.array(signals, dtype=np.float32)
labels = dff[:, 1800]                             #These are their associated class labels (signals)

print(signals.shape)


# In[3]:


df.head()


# In[4]:


df = pd.read_csv("KU-HAR_time_domain_subsamples_20750x300.csv",header=None)


# In[5]:


print(df.shape)


# In[6]:


df.head()


# In[7]:


col_names={}
for i in range(df.shape[1]):
    col_names.update({i: 'c' + str(i)})


# In[8]:


df_col_names = df.rename(columns = col_names)


# In[9]:


df = df_col_names.copy()


# In[10]:


df.describe()


# In[11]:


dff = df.values
signals = dff[:, 0: 1800]                         #These are the time-domian subsamples (signals) 
signals = np.array(signals, dtype=np.float32)
labels = dff[:, 1800]                             #These are their associated class labels (signals)

print(signals.shape)
print(labels.shape)


# In[12]:


# Visualization of the 20001th (time-domain HAR) sample (channel data):
# The correspondig activity is "Jump"

Accelerometer_X_axis_data = signals[20000, 0: 300]
Accelerometer_Y_axis_data = signals[20000, 300: 600]
Accelerometer_Z_axis_data = signals[20000, 600: 900]
Gyroscope_X_axis_data = signals[20000, 900: 1200]
Gyroscope_Y_axis_data = signals[20000, 1200: 1500]
Gyroscope_Z_axis_data = signals[20000, 1500: 1800]
time = np.linspace(.01, 3, 300)

figure(figsize=(30, 30), dpi=80)

ax1 = plt.subplot(611)
ax1.plot(time, Accelerometer_X_axis_data, 'b')
ax1.title.set_text('Accelerometer X axis')
ax1.set_xlabel('time (ms) ->')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.grid(True)


ax2 = plt.subplot(612)
ax2.plot(time, Accelerometer_Y_axis_data, 'g')
ax2.title.set_text('Accelerometer Y axis')
ax2.set_xlabel('time (ms) ->')
ax2.set_ylabel('Acceleration (m/s^2)')
ax2.grid(True)

ax3 = plt.subplot(613)
ax3.plot(time, Accelerometer_Z_axis_data, 'r')
ax3.title.set_text('Accelerometer Z axis')
ax3.set_xlabel('time (ms) ->')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.grid(True)

ax4 = plt.subplot(614)
ax4.plot(time, Gyroscope_X_axis_data, 'b')
ax4.title.set_text('Gyroscope X axis')
ax4.set_xlabel('time (ms) ->')
ax4.set_ylabel('Angular rotation (rad/s)')
ax4.grid(True)

ax5 = plt.subplot(615)
ax5.plot(time, Gyroscope_Y_axis_data, 'g')
ax5.title.set_text('Gyroscope Y axis')
ax5.set_xlabel('time (ms) ->')
ax5.set_ylabel('Angular rotation (rad/s)')
ax5.grid(True)

ax6 = plt.subplot(616)
ax6.plot(time, Gyroscope_Z_axis_data, 'r')
ax6.title.set_text('Gyroscope Z axis')
ax6.set_xlabel('time (ms) ->')
ax6.set_ylabel('Angular rotation (rad/s)')
ax6.grid(True)

plt.show()


# In[13]:


# Discrete Fourier Transform of the time-domain signals, separately for each channel

fft = np.zeros(signals.shape, dtype=np.float32)
for i in range(0,len(signals)):
    for j in range(0, 6):
        tmp = np.fft.fft(signals[i, j*300:(j+1)*300])
        fft[i, j*300:(j+1)*300] = abs(tmp)
        
print(fft.shape)


# In[14]:


pd_fft = pd.DataFrame(fft)
pd_fft.describe()

pd_labels = pd.DataFrame(labels)


# In[15]:


col_names={}
for i in range(pd_fft.shape[1]):
    col_names.update({i: 'c' + str(i)})


# In[16]:


df_col_names = pd_fft.rename(columns = col_names)


# In[17]:


pd_fft = df_col_names.copy()


# In[18]:


pd_fft.head()


# In[19]:


pd_fft.head()


# In[20]:


#pd_fft.to_excel('Transformed data.xlsx')


# In[21]:


fft_freq= np.array(pd_fft, dtype=np.float32)


# In[22]:


# The correspondig activity is "Jump"

Accelerometer_X_axis_data = fft_freq[20000, 0: 300]
Accelerometer_Y_axis_data = fft_freq[20000, 300: 600]
Accelerometer_Z_axis_data = fft_freq[20000, 600: 900]
Gyroscope_X_axis_data = fft_freq[20000, 900: 1200]
Gyroscope_Y_axis_data = fft_freq[20000, 1200: 1500]
Gyroscope_Z_axis_data = fft_freq[20000, 1500: 1800]
time = np.linspace(.01, 3, 300)

figure(figsize=(30, 30), dpi=80)

ax1 = plt.subplot(611)
ax1.plot(time, Accelerometer_X_axis_data, 'b')
ax1.title.set_text('Accelerometer X axis')
ax1.set_xlabel('time (ms) ->')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.grid(True)


ax2 = plt.subplot(612)
ax2.plot(time, Accelerometer_Y_axis_data, 'g')
ax2.title.set_text('Accelerometer Y axis')
ax2.set_xlabel('time (ms) ->')
ax2.set_ylabel('Acceleration (m/s^2)')
ax2.grid(True)

ax3 = plt.subplot(613)
ax3.plot(time, Accelerometer_Z_axis_data, 'r')
ax3.title.set_text('Accelerometer Z axis')
ax3.set_xlabel('time (ms) ->')
ax3.set_ylabel('Acceleration (m/s^2)')
ax3.grid(True)

ax4 = plt.subplot(614)
ax4.plot(time, Gyroscope_X_axis_data, 'b')
ax4.title.set_text('Gyroscope X axis')
ax4.set_xlabel('time (ms) ->')
ax4.set_ylabel('Angular rotation (rad/s)')
ax4.grid(True)

ax5 = plt.subplot(615)
ax5.plot(time, Gyroscope_Y_axis_data, 'g')
ax5.title.set_text('Gyroscope Y axis')
ax5.set_xlabel('time (ms) ->')
ax5.set_ylabel('Angular rotation (rad/s)')
ax5.grid(True)

ax6 = plt.subplot(616)
ax6.plot(time, Gyroscope_Z_axis_data, 'r')
ax6.title.set_text('Gyroscope Z axis')
ax6.set_xlabel('time (ms) ->')
ax6.set_ylabel('Angular rotation (rad/s)')
ax6.grid(True)

plt.show()


# In[23]:


dff = df.values
signals = dff[:, 0: 1800]                         #These are the time-domian subsamples (signals) 
signals = np.array(signals, dtype=np.float32)
labels = dff[:, 1800]                             #These are their associated class labels (signals)

print(signals.shape)
print(labels.shape)


# In[25]:


Accelerometer_X_axis_data


# In[26]:


Accelerometer_Y_axis_data


# In[27]:


Accelerometer_Z_axis_data


# In[28]:


Accelerometer_X_axis_data = np.mean(fft_freq[20000, 0: 300])
Accelerometer_Y_axis_data = np.mean(fft_freq[20000, 300: 600])
Accelerometer_Z_axis_data = np.mean(fft_freq[20000, 600: 900])
Gyroscope_X_axis_data = np.mean(fft_freq[20000, 900: 1200])
Gyroscope_Y_axis_data = np.mean(fft_freq[20000, 1200: 1500])
Gyroscope_Z_axis_data = np.mean(fft_freq[20000, 1500: 1800])
time = np.linspace(.01, 3, 300)

figure(figsize=(30, 30), dpi=80)


# In[29]:


Accelerometer_X_axis_data


# In[30]:


Accelerometer_Y_axis_data


# In[31]:


Accelerometer_Z_axis_data


# In[32]:


# Creating Training and Test subsets with randomly chosen samples:

X_train, X_test, y_train, y_test=train_test_split(fft,labels, test_size=0.3, random_state=0, stratify=labels)
print(X_train.shape)
print(X_test.shape)


# In[33]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
X_train= st_x.fit_transform(X_train)    
X_test= st_x.transform(X_test)  


# In[34]:


X_train


# In[ ]:


get_ipython().system('pip install tslearn')


# In[ ]:


from tslearn.clustering import TimeSeriesKMeans

model = TimeSeriesKMeans(n_clusters=5, metric="dtw",
                         max_iter=3, random_state=0)
model.fit(X_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[61]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=18, random_state=0).fit(X_train)
y_pred = kmeans.predict(X_test)


# In[62]:


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,kmeans.predict(X_test))


# In[63]:


score


# In[78]:


from sklearn.metrics import f1_score,precision_score,recall_score
precision=precision_score(y_test, y_pred, average='weighted')


# In[79]:


precision


# In[80]:


recall=recall_score(y_test, y_pred, average='weighted')


# In[81]:


recall


# In[82]:


f1=f1_score(y_test, y_pred, average='weighted')
f1


# In[38]:


labels = kmeans.fit_predict(X_test)


# In[46]:


labels


# In[39]:


plot_count = 2

fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
fig.suptitle('Clusters')
row_i=0
column_j=0
# For each label there is,
# plots every series with that label
for label in set(labels):
    cluster = []
    for i in range(len(labels)):
            if(labels[i]==label):
                axs[row_i, column_j].plot(X_test[i],c="gray",alpha=0.4)
                cluster.append(X_test[i])
    if len(cluster) > 0:
        axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
    axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
    column_j+=1
    if column_j%plot_count == 0:
        row_i+=1
        column_j=0
        
plt.show()


# In[55]:


score

