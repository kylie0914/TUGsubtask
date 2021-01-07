import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import time


class plot_graph:
    def __init__(self):
        super().__init__()

    def plotPelvis(self,expDateFolder , subjectFolderlist, trialFolderlist, All_Trial, timestamp_sec, pelvis_x, pelvis_y, pelvis_z, saveOption='raw', displayPlot=True, savePlot=True, displayMode='all'):
        if displayPlot == True or savePlot == True:
            for trialIdx in range(len(All_Trial)):
                if savePlot:
                    savePathRoot = "/home/sang/dataset/TUG/arrangedData_LabelSave"
                    TimeInfo = str(time.localtime(time.time()).tm_mon) + "_" + str(time.localtime(time.time()).tm_mday) + "_" + str(time.localtime(
                            time.time()).tm_hour) + "-" + str(time.localtime(time.time()).tm_min) + "-" + str(time.localtime(time.time()).tm_sec)
                    saveDatePath = savePathRoot +  "/" + expDateFolder
                    if os.path.isdir(saveDatePath) is False:  # 폴더 없는 경우 
                        os.mkdir(saveDatePath)
                        
                    saveOptionPath = saveDatePath + "/" + saveOption
                    if os.path.isdir(saveOptionPath) is False: # 폴더 없는 경우 
                            os.mkdir(saveOptionPath)   
                            
                    saveSubPath = saveOptionPath + "/" + subjectFolderlist[trialIdx]
                    if os.path.isdir(saveSubPath) is False: # 폴더 없는 경우 
                        os.mkdir(saveSubPath)                    

                            
                    if saveOption=="raw":
                        fileName = 'Raw_' +subjectFolderlist[trialIdx] + "_"+trialFolderlist[trialIdx]+".png"
                    elif saveOption=="norm":
                        fileName = 'Normalized_' +subjectFolderlist[trialIdx] + "_"+trialFolderlist[trialIdx]+".png"

                    else:
                        AssertionError("Check Save Option(raw, norm, filtered)!!!!!!!")
 
                if(displayMode == 'x'):
                    fig = plt.figure()
                    ax = fig.subplots()
                    title = "Subject: " +  subjectFolderlist[trialIdx ] + ",     Trial: " + trialFolderlist[trialIdx] + ", Pelvis X aixs"
                    ax.set_title(title)
                    ax.plot(
                        timestamp_sec[trialIdx], pelvis_x[trialIdx], 'r--', lw=1, label='pelvis_x')
                    ax.set_xlabel('Time [Sec]')
                    ax.set_ylabel('Side axis(X) of Pelvis [mm]')

                elif(displayMode == 'y'):
                    fig = plt.figure()
                    ax = fig.subplots()
                    title = "Subject: " +  subjectFolderlist[trialIdx ] + ",    Trial: " + trialFolderlist[trialIdx] + ", Pelvis Y aixs"
                    ax.set_title(title)
                    ax.plot(
                        timestamp_sec[trialIdx], pelvis_y[trialIdx], 'b--', lw=1, label='pelvis_y')
                    ax.set_xlabel('Time [Sec]')
                    ax.set_ylabel('Vertical axis(Y) of Pelvis [mm]')

                elif(displayMode == 'z'):
                    fig = plt.figure()
                    ax = fig.subplots()
                    title_z = "Subject: " +  subjectFolderlist[trialIdx ] + ",   Trial: " + trialFolderlist[trialIdx] +  ", Pelvis Z aixs"
                    ax.set_title(title_z)
                    ax.plot(
                        timestamp_sec[trialIdx], pelvis_z[trialIdx], 'g--', lw=1, label=' pelvis_z ')
                    ax.set_xlabel('Time [Sec]')
                    ax.set_ylabel('Depth Axis of Pelvis [mm]')

                elif(displayMode == 'all'):
                    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
                    title_x = "Subject: " +  subjectFolderlist[trialIdx ] + ",   Trial: " + trialFolderlist[trialIdx] +  ", Pelvis X aixs"
                    title_y = "Subject: " +  subjectFolderlist[trialIdx ] + ",   Trial: " + trialFolderlist[trialIdx] +  ", Pelvis Y aixs"
                    title_z = "Subject: " +  subjectFolderlist[trialIdx ] + ",   Trial: " + trialFolderlist[trialIdx] +  ", Pelvis Z aixs"
                    ax[0].set_title(title_x)
                    ax[1].set_title(title_y)
                    ax[2].set_title(title_z)  # position=(1.0+0.07, 1.0)

                    # ax[0].set_xlabel('Time [Sec]')
                    ax[0].set_ylabel('Side axis(X) of Pelvis [mm]')
                    # ax[1].set_xlabel('Time [Sec]')
                    ax[1].set_ylabel('Vertical axis(Y) of Pelvis [mm]')
                    ax[2].set_ylabel('Depth Axis of Pelvis [mm]')
                    ax[2].set_xlabel('Time [Sec]')

                    ax[0].plot(timestamp_sec[trialIdx],
                               pelvis_x[trialIdx], 'r--', lw=1, label='pelvis_x')
                    ax[1].plot(timestamp_sec[trialIdx],
                               pelvis_y[trialIdx], 'b--', lw=1, label='pelvis_y')
                    ax[2].plot(timestamp_sec[trialIdx],
                               pelvis_z[trialIdx], 'g--', lw=1, label='pelvis_z')
                else:
                    AssertionError(
                        "Can't not plot-----! chec}k sensor arguments---!")

                fig.legend()
                plt.show()
                plt.pause(0.03)
                if savePlot:
                    fig.savefig(saveSubPath + "/" + fileName)
                    print("DateFolder: {0} , sujbecFolder: {1} , TrialFolder: {2}".format(expDateFolder, subjectFolderlist[trialIdx], trialFolderlist[trialIdx]))
                plt.close()

    def plot_actionSplit(self,expDateFolder, subjectFolderlist, trialFolderlist, All_Trial, timestamp_sec, pelvis_x, pelvis_y, pelvis_z, actionIdx, displayPlot=True, savePlot=False, saveOption ='actionImg'):
        if displayPlot == True or savePlot == True:                
            for trialIdx in range(len(All_Trial)):
                print(All_Trial[trialIdx])
                print(actionIdx[0][trialIdx],actionIdx[1][trialIdx],actionIdx[2][trialIdx],actionIdx[3][trialIdx],actionIdx[4][trialIdx],actionIdx[5][trialIdx])
                if savePlot:
                    savePathRoot = "/home/sang/dataset/TUG/arrangedData_LabelSave"
                    TimeInfo = str(time.localtime(time.time()).tm_mon) + "_" + str(time.localtime(time.time()).tm_mday) + "_" + str(time.localtime(
                            time.time()).tm_hour) + "-" + str(time.localtime(time.time()).tm_min) + "-" + str(time.localtime(time.time()).tm_sec)
                    saveDatePath = savePathRoot +  "/" + expDateFolder
                    if os.path.isdir(saveDatePath) is False:  # 폴더 없는 경우 
                        os.mkdir(saveDatePath)
                        
                    saveOptionPath = saveDatePath + "/" + saveOption
                    if os.path.isdir(saveOptionPath) is False: # 폴더 없는 경우 
                            os.mkdir(saveOptionPath)   
                            
                    saveSubPath = saveOptionPath + "/" + subjectFolderlist[trialIdx]
                    if os.path.isdir(saveSubPath) is False: # 폴더 없는 경우 
                        os.mkdir(saveSubPath)                    
                        
                    
                    fileName = 'actionWithFilter_' +subjectFolderlist[trialIdx] + "_"+trialFolderlist[trialIdx]+".png"

                fig, ax = plt.subplots(nrows=3, ncols=1,figsize=(12, 10))
                ax[0].plot(timestamp_sec[trialIdx], pelvis_x[trialIdx], 'r-',
                        label='filtered pelvis_y')
                title = "Trial Idx: " + str(trialIdx) + ", acionIdx:"+ str(actionIdx[0][trialIdx]) +","+ str(actionIdx[1][trialIdx]) +","+ str(actionIdx[2][trialIdx]) +","+ str(actionIdx[3][trialIdx]) +","+ str(actionIdx[4][trialIdx]) +","+ str(actionIdx[5][trialIdx])
                ax[0].set_title(title)
                ax[0].set_xlabel('Time [Sec]'); ax[0].set_ylabel('Side Axis of pelvis [mm]')

                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[0][trialIdx]],
                           color='r', linestyle="--", linewidth=3)
                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[5][trialIdx]],
                           color='r', linestyle=":", linewidth=3)
                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[1][trialIdx]],
                           color='g', linestyle="--", linewidth=3)
                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[4][trialIdx] ],
                           color='g', linestyle=":", linewidth=3)
                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[2][trialIdx]],
                           color='k', linestyle="--", linewidth=3)
                ax[0].axvline(x=timestamp_sec[trialIdx][actionIdx[3][trialIdx]],
                           color='k', linestyle=":", linewidth=3)


                ax[1].plot(timestamp_sec[trialIdx], pelvis_y[trialIdx], 'b-',
                        label='filtered pelvis_y')
                ax[1].set_xlabel('Time [Sec]'); ax[1].set_ylabel('Vertical Axis of pelvis [mm]')

                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[0][trialIdx]],
                           color='r', linestyle="--", linewidth=3)
                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[5][trialIdx]],
                           color='r', linestyle=":", linewidth=3)
                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[1][trialIdx]],
                           color='g', linestyle="--", linewidth=3)
                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[4][trialIdx] ],
                           color='g', linestyle=":", linewidth=3)
                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[2][trialIdx]],
                           color='k', linestyle="--", linewidth=3)
                ax[1].axvline(x=timestamp_sec[trialIdx][actionIdx[3][trialIdx]],
                           color='k', linestyle=":", linewidth=3)
                
                ax[2].plot(timestamp_sec[trialIdx], pelvis_z[trialIdx], 'g-',
                        label='filtered pelvis_z')
                ax[2].set_xlabel('Time [Sec]'); ax[2].set_ylabel('Depth axis of pelvis [mm]')

                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[0][trialIdx]],
                           color='r', linestyle="--", linewidth=3, label = 'move Start (end sitting)')
                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[5][trialIdx]],
                           color='r', linestyle=":", linewidth=3, label = 'move End (2nd sitting)')
                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[1][trialIdx]],
                           color='g', linestyle="--", linewidth=3, label = 'start Walk')
                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[4][trialIdx] ],
                           color='g', linestyle=":", linewidth=3, label = 'start Sit')
                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[2][trialIdx]],
                           color='k', linestyle="--", linewidth=3, label = 'turn Start')
                ax[2].axvline(x=timestamp_sec[trialIdx][actionIdx[3][trialIdx]],
                           color='k', linestyle=":", linewidth=3, label = 'turn End')
                
                fig.legend()
                plt.show()
                plt.pause(0.03)
                if savePlot:
                    fig.savefig(saveSubPath + "/" + fileName)
                    print("DateFolder: {0} , sujbecFolder: {1} , TrialFolder: {2}".format(expDateFolder, subjectFolderlist[trialIdx], trialFolderlist[trialIdx]))
                    print("[Frame] moveStart {0}, sit-stand {1}, turnStart {2}, turnEnd {3}, stand-sit {4}, moveEnd {5}".format(actionIdx[0][trialIdx],actionIdx[1][trialIdx],actionIdx[2][trialIdx],actionIdx[3][trialIdx],actionIdx[4][trialIdx],actionIdx[5][trialIdx]))
                plt.close()
                
                
