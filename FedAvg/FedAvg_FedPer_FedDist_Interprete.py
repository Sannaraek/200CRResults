#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os
import hickle as hkl 


# In[ ]:


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


# In[ ]:


# filepath = mainDir + 'savedModels/'+architectureType+'/'+dataSetName+'/'
# os.makedirs(filepath+'trainingStats', exist_ok=True)

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


# In[ ]:


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




