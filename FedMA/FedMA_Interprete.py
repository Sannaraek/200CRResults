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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
euclid = True
asyncTest = False

savedClientModel = 0
showTrainVerbose = 0
segment_size = 128
num_input_channels = 6
learningRate = 0.01
dropout_rate = 0.5
localEpoch = 5
communicationRound = 200
seperateGraph = False


# In[ ]:


# if(dataSetName == 'UCI'):
#     ACTIVITY_LABEL = ['WALKING', 'WALKING_UPSTAIRS','WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
# else:
#     ACTIVITY_LABEL = ['climbingdown', 'climbingup', 'jumping','lying', 'running', 'sitting', 'standing', 'walking']
# activityCount = len(ACTIVITY_LABEL)

# if(modelType == "DNN"):
#     architectureType = str(algorithm)+'_'+str(learningRate)+'LR_'+str(localEpoch)+'LE_'+str(communicationRound)+'CR_400D_100D_'+str(dataConfig)+'_'+str(optimizer)+'_MULTI_GRAPH'
# else: 
#     architectureType = str(algorithm)+'_'+str(learningRate)+'LR_'+str(localEpoch)+'LE_'+str(communicationRound)+'CR_196-16C_4M_1024D_'+str(dataConfig)+'_'+str(optimizer)+'_MULTI_GRAPH'
# mainDir = ''
# filepath = mainDir + 'savedModels/'+architectureType+'/'+dataSetName+'/'
# os.makedirs(filepath, exist_ok=True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# if(dataSetName=='UCI'):
#     clientCount = 5
# else:
#     clientCount = 15
# print("Num GPUs Available: ", len(
# tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


# filepath = mainDir + 'savedModels/'+architectureType+'/'+dataSetName+'/'
# os.makedirs('./trainingStats', exist_ok=True)

trainLossHistory = hkl.load("trainingStats/clientLossTrain.hkl" )
trainAccHistory = hkl.load( "trainingStats/clientAccuracyTrain.hkl" )
stdTrainLossHistory = hkl.load( "trainingStats/clientLossStdTrain.hkl" )
stdTrainAccHistory = hkl.load( "trainingStats/clientAccuracyStdTrain.hkl" )

testLossHistory = hkl.load( "trainingStats/clientLoss.hkl" )
testAccHistory = hkl.load( "trainingStats/clientAccuracy.hkl" )
stdTestLossHistory = hkl.load( "trainingStats/clientLossStd.hkl" )
stdTestAccHistory = hkl.load( "trainingStats/clientAccuracyStd.hkl" )


if(fastMode == False):
    clientStdTrainLossHistory = hkl.load( "trainingStats/clientAllLossStdTrain.hkl" )
    clientStdTrainAccHistory = hkl.load( "trainingStats/clientAllAccuracyStdTrain.hkl" )
    clientStdTestLossHistory = hkl.load( "trainingStats/clientAllLossStd.hkl" )
    clientStdTestAccHistory = hkl.load( "trainingStats/clientAllAccuracyStd.hkl" )

    clientTrainLossHistory = hkl.load( "trainingStats/clientAllLossTrain.hkl" )
    clientTrainAccHistory = hkl.load( "trainingStats/clientAllAccuracyTrain.hkl" )
    clientTestLossHistory = hkl.load( "trainingStats/clientAllLoss.hkl" )
    clientTestAccHistory = hkl.load( "trainingStats/clientAllAccuracy.hkl" )

if(algorithm != 'FEDPER'):
    serverTrainLossHistory = hkl.load( "trainingStats/serverLossTrain.hkl" )
    serverTrainAccHistory = hkl.load( "trainingStats/serverAccuracyTrain.hkl" )
    serverTestLossHistory = hkl.load( "trainingStats/serverLoss.hkl" )
    serverTestAccHistory = hkl.load( "trainingStats/serverAccuracy.hkl" )


# In[ ]:


def saveGraph(title = "",accuracyOrLoss = "Accuracy",asyTest = False,legendLoc = 'lower right'):
    if(asyTest):
        for stage in range(len(roundEnd)):
            plt.axvline(roundEnd[stage], 0, 1,color ="blue")
    plt.title(title)
    plt.ylabel(accuracyOrLoss)
    plt.xlabel('Communication Round')
    plt.legend(loc=legendLoc)
    plt.savefig(title.replace(" ", "")+'.png', dpi=100)
    plt.clf()


# In[ ]:


epoch_range = range(1, communicationRound+1)
if(seperateGraph):
    if(algorithm != "FEDPER"):
        plt.plot(epoch_range, serverTrainAccHistory, label = 'Server Train')
        plt.plot(epoch_range, serverTestAccHistory, label= 'Server Test')
        plt.plot(epoch_range, serverTrainAccHistory,markevery=[np.argmax(serverTrainAccHistory)], ls="", marker="o",color="blue")
        plt.plot(epoch_range, serverTestAccHistory,markevery=[np.argmax(serverTestAccHistory)], ls="", marker="o",color="orange")
        saveGraph("Server accuracy","Accuracy",asyTest = asyncTest)
        
    plt.errorbar(epoch_range, trainAccHistory, yerr=stdTrainAccHistory, label='Client Own Train',alpha=0.6)
    plt.errorbar(epoch_range, testAccHistory, yerr=stdTestAccHistory, label='Client Own Test',alpha=0.6)
    plt.plot(epoch_range, trainAccHistory,markevery=[np.argmax(trainAccHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testAccHistory,markevery=[np.argmax(testAccHistory)], ls="", marker="o",color="red")  
    saveGraph("Client own accuracy","Accuracy",asyTest = asyncTest)


    if(fastMode == False):
        plt.errorbar(epoch_range, clientTrainAccHistory, yerr=clientStdTrainAccHistory, label='Client All Train',alpha=0.6)
        plt.errorbar(epoch_range, clientTestAccHistory, yerr=clientStdTestAccHistory, label='Client All Test',alpha=0.6)
        plt.plot(epoch_range, clientTrainAccHistory,markevery=[np.argmax(clientTrainAccHistory)], ls="", marker="o",color="purple")
        plt.plot(epoch_range, clientTestAccHistory,markevery=[np.argmax(clientTestAccHistory)], ls="", marker="o",color="brown")  
        saveGraph("Client all accuracy","Accuracy",asyTest = asyncTest)

    if(algorithm != "FEDPER"):
        plt.plot(epoch_range, serverTrainLossHistory, label = 'Server Train')
        plt.plot(epoch_range, serverTestLossHistory, label= 'Server Test')
        plt.plot(epoch_range, serverTrainLossHistory,markevery=[np.argmax(serverTrainLossHistory)], ls="", marker="o",color="blue")
        plt.plot(epoch_range, serverTestLossHistory,markevery=[np.argmax(serverTestLossHistory)], ls="", marker="o",color="orange") 
        saveGraph("Server loss","Loss",asyTest = asyncTest,legendLoc = 'upper right')


    plt.errorbar(epoch_range, trainLossHistory, yerr=stdTrainLossHistory, label='Client Own Train',alpha=0.6)
    plt.errorbar(epoch_range, testLossHistory, yerr=stdTestLossHistory, label='Client Own Test',alpha=0.6)
    plt.plot(epoch_range, trainLossHistory,markevery=[np.argmax(trainLossHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testLossHistory,markevery=[np.argmax(testLossHistory)], ls="", marker="o",color="red") 
    
    saveGraph("Client own loss","Loss",asyTest = asyncTest,legendLoc = 'upper right')



    if(fastMode == False):
        plt.errorbar(epoch_range, clientTrainLossHistory, yerr=clientStdTrainLossHistory, label='Client All Train',alpha=0.6)
        plt.errorbar(epoch_range, clientTestLossHistory, yerr=clientStdTestLossHistory, label='Client All Test',alpha=0.6)
        plt.plot(epoch_range, clientTrainLossHistory,markevery=[np.argmax(clientTrainLossHistory)], ls="", marker="o",color="purple")
        plt.plot(epoch_range, clientTestLossHistory,markevery=[np.argmax(clientTestLossHistory)], ls="", marker="o",color="brown")  
        saveGraph("Client all loss","Loss",asyTest = asyncTest,legendLoc = 'upper right')
else:
    if(algorithm != "FEDPER"):
        plt.plot(epoch_range, serverTrainAccHistory, label = 'Server Train')
        plt.plot(epoch_range, serverTestAccHistory, label= 'Server Test')
        plt.plot(epoch_range, serverTrainAccHistory,markevery=[np.argmax(serverTrainAccHistory)], ls="", marker="o",color="blue")
        plt.plot(epoch_range, serverTestAccHistory,markevery=[np.argmax(serverTestAccHistory)], ls="", marker="o",color="orange") 

    plt.errorbar(epoch_range, trainAccHistory, yerr=stdTrainAccHistory, label='Client Own Train',alpha=0.6, color= "green")
    plt.errorbar(epoch_range, testAccHistory, yerr=stdTestAccHistory, label='Client Own Test',alpha=0.6, color='red')

    plt.plot(epoch_range, trainAccHistory,markevery=[np.argmax(trainAccHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testAccHistory,markevery=[np.argmax(testAccHistory)], ls="", marker="o",color="red")  

    # if(asyncTest):
    #     plt.axvline(roundEnd[0], 0, 1,color ="blue")
    #     plt.axvline(roundEnd[1], 0, 1,color ="blue")

    if(fastMode == False):
        plt.errorbar(epoch_range, clientTrainAccHistory, yerr=clientStdTrainAccHistory, label='Client All Train',alpha=0.6, color="purple")
        plt.errorbar(epoch_range, clientTestAccHistory, yerr=clientStdTestAccHistory, label='Client All Test',alpha=0.6, color="brown")
        plt.plot(epoch_range, clientTrainAccHistory,markevery=[np.argmax(clientTrainAccHistory)], ls="", marker="o",color="purple")
        plt.plot(epoch_range, clientTestAccHistory,markevery=[np.argmax(clientTestAccHistory)], ls="", marker="o",color="brown")  

        
    if(asyncTest):
        for stage in range(len(roundEnd)):
            plt.axvline(roundEnd[stage], 0, 1,color ="blue")
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

    plt.errorbar(epoch_range, trainLossHistory, yerr=stdTrainLossHistory, label='Client Own Train',alpha=0.6, color='green')
    plt.errorbar(epoch_range, testLossHistory, yerr=stdTestLossHistory, label='Client Own Test',alpha=0.6, color='red')
    plt.plot(epoch_range, trainLossHistory,markevery=[np.argmin(trainLossHistory)], ls="", marker="o",color="green")
    plt.plot(epoch_range, testLossHistory,markevery=[np.argmin(testLossHistory)], ls="", marker="o",color="red")  



    if(fastMode == False):
        plt.errorbar(epoch_range, clientTrainLossHistory, yerr=clientStdTrainLossHistory, label='Client All Train',alpha=0.6,color="purple")
        plt.errorbar(epoch_range, clientTestLossHistory, yerr=clientStdTestLossHistory, label='Client All Test',alpha=0.6,color="brown")
        plt.plot(epoch_range, clientTrainLossHistory,markevery=[np.argmin(clientTrainLossHistory)], ls="", marker="o",color="purple")
        plt.plot(epoch_range, clientTestLossHistory,markevery=[np.argmin(clientTestLossHistory)], ls="", marker="o",color="brown")  




    if(asyncTest):
        for stage in range(len(roundEnd)):
            plt.axvline(roundEnd[stage], 0, 1,color ="blue")
#         plt.axvline(roundEnd[1], 0, 1,color ="blue")

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Communication Round')
    plt.legend(loc= 'upper right')
    plt.savefig('LearningLoss.png', dpi=100)
    plt.show()


# In[ ]:




