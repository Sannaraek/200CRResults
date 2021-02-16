#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv


import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import time
from tensorflow.python.client import device_lib
import tensorflow as tf
import os
import hickle as hkl 
import skbio as skb
from scipy.spatial import distance_matrix



# In[6]:


np.random.seed(0)
# which GPU to use
# "-1,0,1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# DNN,CNN
modelType = "CNN"

# algorithm = "FEDAVG,FEDPER"
algorithm = "FEDAVG"

# UCI,REALWORLD_CLIENT
dataSetName = 'REALWORLD_CLIENT'

#BALANCED, UNBALANCED
dataConfig = "BALANCED"

#ADAM, SGD
optimizer = "SGD"

#0, 1
mantel = False
fastMode = False
savedClientModel = 0
showTrainVerbose = 0
segment_size = 128
num_input_channels = 6
learningRate = 0.01
dropout_rate = 0.5
localEpoch = 5
communicationRound = 200
asyncTest = False


# In[3]:


if(dataSetName == 'UCI'):
    ACTIVITY_LABEL = ['WALKING', 'WALKING_UPSTAIRS','WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
else:
    ACTIVITY_LABEL = ['climbingdown', 'climbingup', 'jumping','lying', 'running', 'sitting', 'standing', 'walking']
activityCount = len(ACTIVITY_LABEL)

if(modelType == "DNN"):
    architectureType = str(algorithm)+'_'+str(learningRate)+'LR_'+str(localEpoch)+'LE_'+str(communicationRound)+'CR_400D_100D_'+str(dataConfig)+'_'+str(optimizer)+'_MULTI_GRAPH'
else: 
    architectureType = str(algorithm)+'_'+str(learningRate)+'LR_'+str(localEpoch)+'LE_'+str(communicationRound)+'CR_196-16C_4M_1024D_'+str(dataConfig)+'_'+str(optimizer)+'_MULTI_GRAPH'
mainDir = ''
filepath = mainDir + 'savedModels/'+architectureType+'/'+dataSetName+'/'
os.makedirs(filepath, exist_ok=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
if(dataSetName=='UCI'):
    clientCount = 5
else:
    clientCount = 15
print("Num GPUs Available: ", len(
tf.config.experimental.list_physical_devices('GPU')))


# In[4]:


# filepath = mainDir + 'savedModels/'+architectureType+'/'+dataSetName+'/'
os.makedirs(filepath+'trainingStats', exist_ok=True)

trainLossHistory = hkl.load("trainingStats/trainLossHistory.hkl" )
trainAccHistory = hkl.load( "trainingStats/trainAccHistory.hkl" )
stdTrainLossHistory = hkl.load( "trainingStats/stdTrainLossHistory.hkl" )
stdTrainAccHistory = hkl.load( "trainingStats/stdTrainAccHistory.hkl" )

testLossHistory = hkl.load( "trainingStats/testLossHistory.hkl" )
testAccHistory = hkl.load( "trainingStats/testAccHistory.hkl" )
stdTestLossHistory = hkl.load( "trainingStats/stdTestLossHistory.hkl" )
stdTestAccHistory = hkl.load( "trainingStats/stdTestAccHistory.hkl" )


if(fastMode == False):
    clientStdTrainLossHistory = hkl.load( "trainingStats/clientStdTrainLossHistory.hkl" )
    clientStdTrainAccHistory = hkl.load( "trainingStats/clientStdTrainAccHistory.hkl" )
    clientStdTestLossHistory = hkl.load( "trainingStats/clientStdTestLossHistory.hkl" )
    clientStdTestAccHistory = hkl.load( "trainingStats/clientStdTestAccHistory.hkl" )

    clientTrainLossHistory = hkl.load( "trainingStats/clientTrainLossHistory.hkl" )
    clientTrainAccHistory = hkl.load( "trainingStats/clientTrainAccHistory.hkl" )
    clientTestLossHistory = hkl.load( "trainingStats/clientTestLossHistory.hkl" )
    clientTestAccHistory = hkl.load( "trainingStats/clientTestAccHistory.hkl" )

if(algorithm != 'FEDPER'):
    serverTrainLossHistory = hkl.load( "trainingStats/serverTrainLossHistory.hkl" )
    serverTrainAccHistory = hkl.load( "trainingStats/serverTrainAccHistory.hkl" )
    serverTestLossHistory = hkl.load( "trainingStats/serverTestLossHistory.hkl" )
    serverTestAccHistory = hkl.load( "trainingStats/serverTestAccHistory.hkl" )


# In[10]:


epoch_range = range(1, communicationRound+1)


if(algorithm != "FEDPER"):
    plt.plot(epoch_range, serverTrainAccHistory, label = 'Server Train')
    plt.plot(epoch_range, serverTestAccHistory, label= 'Server Test')
    plt.plot(epoch_range, serverTrainAccHistory,markevery=[np.argmax(serverTrainAccHistory)], ls="", marker="o",color="blue")
    plt.plot(epoch_range, serverTestAccHistory,markevery=[np.argmax(serverTestAccHistory)], ls="", marker="o",color="orange") 

plt.errorbar(epoch_range, trainAccHistory, yerr=stdTrainAccHistory, label='Client Own Train',alpha=0.6)
plt.errorbar(epoch_range, testAccHistory, yerr=stdTestAccHistory, label='Client Own Test',alpha=0.6)

if(asyncTest):
    plt.axvline(roundEnd[0], 0, 1,color ="blue")
    plt.axvline(roundEnd[1], 0, 1,color ="blue")

if(fastMode == False):
    plt.errorbar(epoch_range, clientTrainAccHistory, yerr=clientStdTrainAccHistory, label='Client All Train',alpha=0.6)
    plt.errorbar(epoch_range, clientTestAccHistory, yerr=clientStdTestAccHistory, label='Client All Test',alpha=0.6)
    plt.plot(epoch_range, clientTrainAccHistory,markevery=[np.argmax(clientTrainAccHistory)], ls="", marker="o",color="purple")
    plt.plot(epoch_range, clientTestAccHistory,markevery=[np.argmax(clientTestAccHistory)], ls="", marker="o",color="brown")  



plt.plot(epoch_range, trainAccHistory,markevery=[np.argmax(trainAccHistory)], ls="", marker="o",color="green")
plt.plot(epoch_range, testAccHistory,markevery=[np.argmax(testAccHistory)], ls="", marker="o",color="red")  




plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Communication Round')
plt.legend(loc='lower right')
plt.savefig('LearningAccuracy.png', dpi=100)
plt.show()

if(algorithm != "FEDPER"):
    plt.plot(epoch_range, serverTrainLossHistory, label = 'Server Train')
    plt.plot(epoch_range, serverTestLossHistory, label= 'Server Test')
    plt.plot(epoch_range, serverTrainLossHistory,markevery=[np.argmin(serverTrainLossHistory)], ls="", marker="o",color="blue")
    plt.plot(epoch_range, serverTestLossHistory,markevery=[np.argmin(serverTestLossHistory)], ls="", marker="o",color="orange") 

plt.errorbar(epoch_range, trainLossHistory, yerr=stdTrainLossHistory, label='Client Own Train',alpha=0.6)
plt.errorbar(epoch_range, testLossHistory, yerr=stdTestLossHistory, label='Client Own Test',alpha=0.6)

if(fastMode == False):
    plt.errorbar(epoch_range, clientTrainLossHistory, yerr=clientStdTrainLossHistory, label='Client All Train',alpha=0.6)
    plt.errorbar(epoch_range, clientTestLossHistory, yerr=clientStdTestLossHistory, label='Client All Test',alpha=0.6)
    plt.plot(epoch_range, clientTrainLossHistory,markevery=[np.argmin(clientTrainLossHistory)], ls="", marker="o",color="purple")
    plt.plot(epoch_range, clientTestLossHistory,markevery=[np.argmin(clientTestLossHistory)], ls="", marker="o",color="brown")  



plt.plot(epoch_range, trainLossHistory,markevery=[np.argmin(trainLossHistory)], ls="", marker="o",color="green")
plt.plot(epoch_range, testLossHistory,markevery=[np.argmin(testLossHistory)], ls="", marker="o",color="red")  


if(asyncTest):
    plt.axvline(roundEnd[0], 0, 1,color ="blue")
    plt.axvline(roundEnd[1], 0, 1,color ="blue")

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Communication Round')
plt.legend(loc= 'upper right')
plt.savefig('LearningLoss.png', dpi=100)
plt.show()


# In[ ]:


# epoch_range = range(1, communicationRound+1)

# plt.plot(epoch_range, serverTrainAccHistory)
# plt.plot(epoch_range, serverTestAccHistory)

# # powderblue
# # bisque

# plt.title('Server accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Communication Round')
# plt.legend(['Server Train','Server Test'], loc='lower right')
# plt.savefig('LearningAccuracyServer.png', dpi=100)
# plt.clf()

# plt.plot(epoch_range, trainAccHistory)
# plt.plot(epoch_range, testAccHistory)

# plt.fill_between(epoch_range, trainAccHistory - stdTrainAccHistory, trainAccHistory + stdTrainAccHistory, color="powderblue",alpha=0.5)
# plt.fill_between(epoch_range, testAccHistory - stdTestAccHistory, testAccHistory + stdTestAccHistory, color="bisque",alpha=0.5)

# plt.title('Clients accuracy on own dataset')
# plt.ylabel('Accuracy')
# plt.xlabel('Communication Round')
# if(algorithm != "FEDPER"):
#     plt.legend(['Client Train','Client Test',], loc='lower right')
# else:
#     plt.legend(['Client Train','Client Test'], loc='lower right')
    
# plt.savefig('LearningAccuracyClientSingle.png', dpi=100)
# plt.clf()

# if(fastMode == 0):
#     plt.plot(epoch_range, clientTrainAccHistory)
#     plt.plot(epoch_range, clientTestAccHistory)

#     plt.fill_between(epoch_range, clientTrainAccHistory - clientStdTrainAccHistory, clientTrainAccHistory + clientStdTrainAccHistory, color="powderblue",alpha=0.5)
#     plt.fill_between(epoch_range, clientTestAccHistory - clientStdTestAccHistory, clientTestAccHistory + clientStdTestAccHistory, color="bisque",alpha=0.5)

#     plt.title('Clients accuracy on all dataset')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Communication Round')
#     if(algorithm != "FEDPER"):
#         plt.legend(['Client Train','Client Test',], loc='lower right')
#     else:
#         plt.legend(['Client Train','Client Test'], loc='lower right')

#     plt.savefig('LearningAccuracyClientAll.png', dpi=100)
#     plt.clf()


# In[ ]:


# epoch_range = range(1, communicationRound+1)

# plt.plot(epoch_range, serverTrainLossHistory)
# plt.plot(epoch_range, serverTestLossHistory)

# # powderblue
# # bisque

# plt.title('Server loss')
# plt.ylabel('Loss')
# plt.xlabel('Communication Round')
# plt.legend(['Server Train','Server Test'], loc='lower right')
# plt.savefig('LearningLossServer.png', dpi=100)
# plt.clf()

# plt.plot(epoch_range, trainLossHistory)
# plt.plot(epoch_range, testLossHistory)

# plt.fill_between(epoch_range, trainLossHistory - stdTrainLossHistory, trainLossHistory + stdTrainLossHistory, color="powderblue",alpha=0.5)
# plt.fill_between(epoch_range, testLossHistory - stdTestLossHistory, testLossHistory + stdTestLossHistory, color="bisque",alpha=0.5)

# plt.title('Clients loss on own dataset')
# plt.ylabel('Loss')
# plt.xlabel('Communication Round')
# if(algorithm != "FEDPER"):
#     plt.legend(['Client Train','Client Test',], loc='lower right')
# else:
#     plt.legend(['Client Train','Client Test'], loc='lower right')

# plt.savefig('LearningLossClientSingle.png', dpi=100)
# plt.clf()

# if(fastMode == 0):
#     plt.plot(epoch_range, clientTrainLossHistory)
#     plt.plot(epoch_range, clientTestLossHistory)

#     plt.fill_between(epoch_range, clientTrainLossHistory - clientStdTrainLossHistory, clientTrainLossHistory + clientStdTrainLossHistory, color="powderblue",alpha=0.5)
#     plt.fill_between(epoch_range, clientTestLossHistory - clientStdTestLossHistory, clientTestLossHistory + clientStdTestLossHistory, color="bisque",alpha=0.5)

#     plt.title('Clients loss on all dataset')
#     plt.ylabel('Loss')
#     plt.xlabel('Communication Round')
#     if(algorithm != "FEDPER"):
#         plt.legend(['Client Train','Client Test',], loc='lower right')
#     else:
#         plt.legend(['Client Train','Client Test'], loc='lower right')

#     plt.savefig('LearningLossClientAll.png', dpi=100)
#     plt.clf()


# In[ ]:




