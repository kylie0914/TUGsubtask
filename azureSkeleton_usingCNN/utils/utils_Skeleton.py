import os
import copy,codecs
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt       # for LPF, HPF
from scipy.signal import find_peaks, peak_prominences  # for peak detection

class utilsSkeleton:
    def __init__(self):
        super().__init__()  # overriding 금지 (상속 안해서 상관 없긴 한데..)
        print("utils skeleton 객체 생성 -- Initialize")
        self.eachTrialDir = []  # subjectFoloderlist
        self.subjectFolderlist = []
        self.trialFolderlist =[]
        self.csvFilelist = []
        self.totalTrials = 0

        self.safeTimeforBody = 66
        self.oneHot_action = []
        
        self.timestamp_sec = []
        self.rawPelvis_x = []
        self.rawPelvis_y = []
        self.rawPelvis_z = []
        
        self.normPelvis_x = []
        self.normPelvis_y = []
        self.normPelvis_z = []
        
        self.lpfPelvis_x = []
        self.lpfPelvis_y = []
        self.lpfPelvis_z = []
        
        self.sitting = []
        self.stand_sit = []
        self.walking = []
        self.turning = []
        self.stand_sit = []

    def getDatalist_forDateFolder(self, ROOT_DIR, expDateFolder, displayList=False):
        dateFolder_DIR = ROOT_DIR+"/"+expDateFolder

        for subjectIndex, subjectName in enumerate(sorted(os.listdir(dateFolder_DIR))):
            subjectFolder_DIR = dateFolder_DIR + "/" + subjectName

            if displayList:
                print("     ---------------------------------------------------------------------------------------------------------------------------------         ")

            for trialIndex, trialName in enumerate(sorted(os.listdir(subjectFolder_DIR))):
                if fnmatch.fnmatch(trialName, "*.csv"):
                    continue
                else:
                    trialFolder_DIR = subjectFolder_DIR + "/" + trialName
                    os.chdir(trialFolder_DIR)  # 작업  path 변경

                    self.eachTrialDir.append(trialFolder_DIR)
                    self.subjectFolderlist.append(subjectName)
                    self.trialFolderlist.append(trialName)

                    for (dir_path, dir_folder, files) in os.walk(trialFolder_DIR):
                        csvlist = []
                        if len(files) > 0:
                            for fileName in sorted(files):
                                if fnmatch.fnmatch(fileName, "*.csv"):
                                    # csvlist.append(fileName)
                                    self.totalTrials += 1  # of colorImg
                                    self.csvFilelist.append(fileName)

                if displayList:
                    print("     Exp.Date: {0},      Subject: {1} ,      # of Trials per subject: {2} ,       Current Trial: {3},     Trial: {4}".format(expDateFolder, subjectName, trialIndex, self.totalTrials, trialName))
        # self.totalTrials += 1
        os.chdir(ROOT_DIR)
        print("-----> [ Total Trials] (# of Trial): ", self.totalTrials)


    def getRawData_forDateFolder(self, ROOT_DIR, expDateFolder, displayTimediff=False):   
        expDateDir = ROOT_DIR + expDateFolder
        for trialIdx in range(len(self.eachTrialDir)):
            os.chdir(self.eachTrialDir[trialIdx])   # print(self.eachTrialDir[trialIdx])
            csvFile = self.csvFilelist[trialIdx]
            
            
            rawData = np.loadtxt(csvFile, delimiter=",", skiprows=1, usecols=[0, 1, 2, 3, 97, 98, 99, 100], encoding='utf-8')

            
            #--------------  Get Pelvis data
            frameNum = rawData[:, 0]
            eachrawPelvis_x = rawData[:, 1]
            eachrawPelvis_y = rawData[:, 2]  # vertical axis = y
            eachrawPelvis_z = rawData[:, 3]
            
            #--------------  각 data 획득 간, 소요 시간 (differential time betwwen two data)
            timestamp_diff_msec = np.array([eachrawPelvis_x])
            for i in range(len(frameNum)):
                before_time_msec = rawData[i-1, 4] * 60 * 60 * 1000 + rawData[i-1, 5] * 60 * 1000 +  rawData[i-1, 6] * 1000 + rawData[i-1, 7] 
                current_time_msec = rawData[i, 4] * 60 * 60 * 1000 + rawData[i, 5] * 60 * 1000 + rawData[i, 6] * 1000 + rawData[i, 7]
                if i is 0:
                    timestamp_diff_msec[0, i] = 0
                else:
                    timestamp_diff_msec[0, i] = int(
                        current_time_msec - before_time_msec)

                    # check body -- Error check
                    if timestamp_diff_msec[0, i] > self.safeTimeforBody:
                        AssertionError("[Lose Body,No data] Can not export Data ")
            if displayTimediff:
                print("Time diff \n ",timestamp_diff_msec )
                
            #-------------- timestamp_sec 계산 ( time diff msec -> timestamp_sec )
            time_sum_msec = 0
            eachTimestamp_sec = np.array(frameNum, dtype=float) # sec (?,1)
            for timeIdx in range(timestamp_diff_msec.shape[1]):
                time_sum_msec = int(time_sum_msec + timestamp_diff_msec[0, timeIdx])
                eachTimestamp_sec[timeIdx] = time_sum_msec / 1000           
                
            self.timestamp_sec.append(eachTimestamp_sec)
            self.rawPelvis_x.append(eachrawPelvis_x)
            self.rawPelvis_y.append(eachrawPelvis_y)
            self.rawPelvis_z.append(eachrawPelvis_z)  
            
            os.chdir(ROOT_DIR)
        else:
            AssertionError("image Index is not in a range")
  
        
    def getNormData_forDateFolder(self, normalizationMethod = 'min_max', displayVal = False):
        self.normPelvis_x = copy.deepcopy(self.rawPelvis_x)
        self.normPelvis_y = copy.deepcopy(self.rawPelvis_y)
        self.normPelvis_z = copy.deepcopy(self.rawPelvis_z)
        
        for trialIdx in range(len(self.eachTrialDir)): 
            if normalizationMethod == 'min_max':
                self.normPelvis_x[trialIdx] = ((self.rawPelvis_x[trialIdx] - np.min(self.rawPelvis_x[trialIdx])) /( np.max(self.rawPelvis_x[trialIdx]) - np.min(self.rawPelvis_x[trialIdx])))
                self.normPelvis_y[trialIdx] = ((self.rawPelvis_y[trialIdx] - np.min(self.rawPelvis_y[trialIdx])) /( np.max(self.rawPelvis_y[trialIdx]) - np.min(self.rawPelvis_y[trialIdx])))
                self.normPelvis_z[trialIdx] = ( (self.rawPelvis_z[trialIdx] - np.min(self.rawPelvis_z[trialIdx])) /( np.max(self.rawPelvis_z[trialIdx]) - np.min(self.rawPelvis_z[trialIdx])))
                
                if displayVal:
                    print("[min_max each axes]\n {0}\n {1}\n{2}".format(self.normPelvis_x[trialIdx],self.normPelvis_y[trialIdx], self.normPelvis_z[trialIdx]) )
            elif normalizationMethod =='z_score':
                self.normPelvis_x[trialIdx] = (self.rawPelvis_x[trialIdx] - np.mean(self.rawPelvis_x[trialIdx])) /np.std(self.rawPelvis_x[trialIdx])
                self.normPelvis_y[trialIdx] = (self.rawPelvis_y[trialIdx] - np.mean(self.rawPelvis_y[trialIdx])) /np.std(self.rawPelvis_y[trialIdx])
                self.normPelvis_z[trialIdx] = (self.rawPelvis_z[trialIdx] - np.mean(self.rawPelvis_z[trialIdx])) /np.std(self.rawPelvis_z[trialIdx])
                
                if displayVal:
                    print("[z_score each axes]\n {0}\n {1}\n{2}".format(self.normPelvis_x[trialIdx],self.normPelvis_y[trialIdx], self.normPelvis_z[trialIdx]) )
            else:
                AssertionError("-------- Choose Normalization Method --------")



    #------------------- Split Action sessions              
    def find_turningIdx(self, dataType='raw', turningPercent=0.95, displayResult=False):  # 이미지로 보니 10%
        turnStartIdx = []
        turnEndIdx = []
        if dataType == 'raw':
            pelvis_x = self.rawPelvis_x
            pelvis_y = self.rawPelvis_y
            pelvis_z = self.rawPelvis_z
                     
        elif dataType == 'norm':
            pelvis_x = self.normPelvis_x
            pelvis_y = self.normPelvis_y
            pelvis_z = self.normPelvis_z
            
        for trialIdx in range(len(self.eachTrialDir)): 
            farDistIdx = np.argmin(pelvis_x[trialIdx])  # 가장 오목한 부분
            farDist = pelvis_x[trialIdx][farDistIdx]
            vertical_at_farDist = pelvis_y[trialIdx][farDistIdx]

            # 0.05% 에 해당하는 만큼을 min 기준으로 +/- 해줌
            turnThresholdIdx = np.round(
                (len(pelvis_x[trialIdx])/2) * (1 - turningPercent)).astype(np.int)  # 전체 길이 *0.95
            temp_turnStartIdx = farDistIdx - turnThresholdIdx
            temp_turnEndIdx = farDistIdx + turnThresholdIdx

            turnStartIdx.append(temp_turnStartIdx)
            turnEndIdx.append(temp_turnEndIdx)
            if displayResult:
                print("turnStart {0} ,   turnEnd {1}".format(
                    temp_turnStartIdx, temp_turnEndIdx))

                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(self.timestamp_sec[trialIdx],
                        pelvis_y[trialIdx], 'b-', label='Pelvis_y')
                title = dataType + "Y axis with Turn start: " + str(temp_turnStartIdx) + ", Turn End: " + str(temp_turnEndIdx)
                ax.set_title(title)
                ax.set_xlabel('Time [Sec]')
                ax.set_ylabel(
                    'Vertical Axis of Pelvis [mm]')

                ax.axvline(x=self.timestamp_sec[trialIdx][farDistIdx],
                           color='r', linestyle="-", linewidth=3)
                ax.axvline(x=self.timestamp_sec[trialIdx][temp_turnStartIdx],
                           color='g', linestyle="--", linewidth=3)
                ax.axvline(x=self.timestamp_sec[trialIdx][temp_turnEndIdx],
                           color='g', linestyle=":", linewidth=3)

                fig.legend()
                plt.show()

        return turnStartIdx, turnEndIdx
    

    
    
    def LPF_forLabel(self,dataType = 'raw', cutoff_freq=0.9, order=1, displayResult=False, displayMode='filtered'):
        if dataType == 'raw':
            pelvis_x = self.rawPelvis_x
            pelvis_y = self.rawPelvis_y
            pelvis_z = self.rawPelvis_z
            Filtered_x = copy.deepcopy(pelvis_x)
            Filtered_y = copy.deepcopy(pelvis_y)
            Filtered_z = copy.deepcopy(pelvis_z)
                     
        elif dataType == 'norm':
            pelvis_x = self.normPelvis_x
            pelvis_y = self.normPelvis_y
            pelvis_z = self.normPelvis_z
            Filtered_x = copy.deepcopy(pelvis_x)
            Filtered_y = copy.deepcopy(pelvis_y)
            Filtered_z = copy.deepcopy(pelvis_z)
            
        for trialIdx in range(len(self.eachTrialDir)): 
            
            sampling_freq = 30
            number_of_samples = len(pelvis_y[trialIdx])
            normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq

            numerator_coeffs, denominator_coeffs = butter(
                order, normalized_cutoff_freq)

            Filtered_x[trialIdx] = (filtfilt(
                numerator_coeffs, denominator_coeffs, Filtered_x[trialIdx]))  # Phase shift 없음
            Filtered_y[trialIdx] = (
                filtfilt(numerator_coeffs, denominator_coeffs, Filtered_y[trialIdx]))
            Filtered_z[trialIdx] = (
                filtfilt(numerator_coeffs, denominator_coeffs, Filtered_z[trialIdx]))

            if displayResult:
                if displayMode == 'raw':
                    fig_raw, ax_raw = plt.subplots( nrows=3, ncols=1, figsize=(12, 10))
                    ax_raw[0].plot(self.timestamp_sec[trialIdx],pelvis_x[trialIdx], 'b-', label='raw pelvis_X')
                    ax_raw[0].set_title(" Raw pelvis X axis ")

                    ax_raw[1].plot(self.timestamp_sec[trialIdx], pelvis_y[trialIdx], 'r-',
                                   linewidth=1, label='raw pelvis_y')
                    ax_raw[1].set_title(" Raw pelvis Y axis ")

                    ax_raw[2].plot(self.timestamp_sec[trialIdx], pelvis_z[trialIdx], 'g-',
                                   linewidth=1, label='raw pelvis_y')
                    ax_raw[2].set_title(" Raw pelvis Z axis ")
                    ax_raw[2].set_xlabel('Time [Sec]')
                    ax_raw[2].set_ylabel(
                        'Vertical Axis of Pelvis [mm]')
                    fig_raw.legend()
                    plt.show()

                elif displayMode == 'filtered':
                    fig_filter, ax_filter = plt.subplots(
                        nrows=3, ncols=1, figsize=(12, 10))
                    ax_filter[0].plot(
                        self.timestamp_sec[trialIdx], Filtered_x[trialIdx], 'b-', label='filtered pelvis_X')
                    ax_filter[0].set_title(" filtered pelvis X axis ")
                    
                    ax_filter[1].plot(self.timestamp_sec[trialIdx], Filtered_y[trialIdx], 'r-',
                                      linewidth=1, label='filtered pelvis_y')
                    ax_filter[1].set_title(" Raw pelvis Y axis ")

                    ax_filter[2].plot(self.timestamp_sec[trialIdx], Filtered_z[trialIdx], 'g-',
                                      linewidth=1, label='filtered pelvis_z')
                    ax_filter[2].set_title(" filtered pelvis Z axis ")
                    ax_filter[2].set_xlabel('Time [Sec]')
                    ax_filter[2].set_ylabel(
                        'Vertical Axis of Pelvis [mm]')
                    fig_filter.legend()
                    plt.show()
                    
        self.lpfPelvis_x = Filtered_x
        self.lpfPelvis_y = Filtered_y
        self.lpfPelvis_z = Filtered_z
        # return Filtered_x, Filtered_y, Filtered_z
    
    

    def find_walkingIdx(self, displayResult=False):
        startWalkIdx = []
        startSitIdx = []

        pelvis_x = self.lpfPelvis_x
        pelvis_y = self.lpfPelvis_y
        pelvis_z = self.lpfPelvis_z
            
        for trialIdx in range(len(self.eachTrialDir)): 
            diffVal = []
            diffThershold = 16                    
            for dataIdx in range(len(pelvis_y[trialIdx])):
                if dataIdx < diffThershold:
                    diffVal.append(0.)
                elif dataIdx + diffThershold >= len(pelvis_y[trialIdx]):
                    diffVal.append(0.)

                else:
                    diffVal.append(
                        pelvis_y[trialIdx][dataIdx + diffThershold] - pelvis_y[trialIdx][dataIdx])

            tempMoveIdx = []
            tempSitIdx = []
            half_idx = np.argmin(pelvis_x[trialIdx])
            for dataIdx in range(len(pelvis_y[trialIdx][:half_idx])):
                if abs(diffVal[dataIdx]) >= 0.16:   # 11_20 ) = -50, 11_03 =-100:
                    tempMoveIdx.append(dataIdx)
                    
            for dataIdx in range(len(pelvis_y[trialIdx][half_idx:])):  
                if abs(diffVal[dataIdx + half_idx]) >= 0.16: #50:  #  11_20 = 50,   11_03 = 100:
                    tempSitIdx.append(dataIdx + half_idx)
                    
                    
            # for dataIdx in range(len(pelvis_y[trialIdx][:half_idx])):
            #     if diffVal[dataIdx] <= -0.2:   # 11_20 ) = -50, 11_03 =-100:
            #         tempMoveIdx.append(dataIdx)
                    
            # for dataIdx in range(len(pelvis_y[trialIdx][half_idx:])):  
            #     if diffVal[dataIdx + half_idx] >= 0.2: #50:  #  11_20 = 50,   11_03 = 100:
            #         tempSitIdx.append(dataIdx + half_idx)
                    
            tempMoveIdx = np.array(tempMoveIdx)
            tempSitIdx = np.array(tempSitIdx)

            temp_startWalk_idx =  np.max(tempMoveIdx)
            temp_startSitIdx = np.min(tempSitIdx)

            startWalkIdx.append(temp_startWalk_idx)
            startSitIdx.append(temp_startSitIdx)

            if displayResult:
                resultVal = np.array(
                    [startWalkIdx[trialIdx], startSitIdx[trialIdx]])
                curlTime_sec = np.array(
                    [self.timestamp_sec[trialIdx][temp_startWalk_idx], self.timestamp_sec[trialIdx][temp_startSitIdx]])
                curlPelvis = np.array(
                    [pelvis_y[trialIdx][temp_startWalk_idx], pelvis_y[trialIdx][temp_startSitIdx]])

                fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
                ax[0].plot(self.timestamp_sec[trialIdx], pelvis_x[trialIdx],
                           'b-', label='filtered pelvis_X')
                title = "startWalk: " +str(temp_startWalk_idx) + " , Standsit: " + str(temp_startSitIdx)
                ax[0].set_title(" filtered pelvis X axis " + title)

                ax[0].axvline(x=self.timestamp_sec[trialIdx][temp_startWalk_idx],
                              color='r', linestyle="--", linewidth=3, label='start Walk')
                ax[0].axvline(x=self.timestamp_sec[trialIdx][temp_startSitIdx],
                              color='r', linestyle=":", linewidth=3, label='start Sit')

                ax[1].plot(self.timestamp_sec[trialIdx], pelvis_y[trialIdx], 'r-',
                           linewidth=1, label='filtered pelvis_y')
                ax[1].set_title(" filtered pelvis Y axis " + title)
                ax[1].axvline(x=self.timestamp_sec[trialIdx][temp_startWalk_idx],
                              color='r', linestyle="--", linewidth=3)
                ax[1].axvline(x=self.timestamp_sec[trialIdx][temp_startSitIdx],
                              color='r', linestyle=":", linewidth=3)

                ax[2].plot(self.timestamp_sec[trialIdx], pelvis_z[trialIdx], 'g-',
                           linewidth=1, label='filtered pelvis_z')
                ax[2].set_title(" filtered pelvis Z axis " + title)
                ax[2].set_xlabel('Time [Sec]')
                ax[2].set_ylabel(
                    'Vertical Axis of Pelvis [mm]')
                ax[2].axvline(x=self.timestamp_sec[trialIdx][temp_startWalk_idx],
                              color='r', linestyle="--", linewidth=3)
                ax[2].axvline(x=self.timestamp_sec[trialIdx][temp_startSitIdx],
                              color='r', linestyle=":", linewidth=3)
                fig.legend()
                plt.show()
        
        return startWalkIdx, startSitIdx
    
    
    def find_MovingIdx(self,displayResult=True):
        starMoveIdx = []
        moveEndIdx = []
        
        pelvis_x = self.lpfPelvis_x
        pelvis_y = self.lpfPelvis_y
        pelvis_z = self.lpfPelvis_z
        

        for trialIdx in range(len(self.eachTrialDir)): 
            diffVal = []
            diffThershold = 16
            for dataIdx in range(len(pelvis_x[trialIdx])):
                if dataIdx < diffThershold:
                    diffVal.append(0.)
                elif dataIdx + diffThershold >= len(pelvis_x[trialIdx]):
                    diffVal.append(0.)

                else:
                    diffVal.append(
                        pelvis_x[trialIdx][dataIdx + diffThershold] - pelvis_y[trialIdx][dataIdx])
                    

            tempMoveIdx = []
            tempSitIdx = []
            half_idx = np.argmin(pelvis_x[trialIdx])

            for dataIdx in range(len(pelvis_x[trialIdx][:half_idx])):
                if abs(diffVal[dataIdx]) > 0.04 and abs(diffVal[dataIdx]) < 0.1:   # 11_20 ) = -50, 11_03 =-100:
                    tempMoveIdx.append(dataIdx)
                    
            for dataIdx in range(len(pelvis_x[trialIdx][half_idx:])):  
                if abs(diffVal[dataIdx + half_idx]) > 0.04 and abs(diffVal[dataIdx]) < 0.1 : #50:  #  11_20 = 50,   11_03 = 100:
                    tempSitIdx.append(dataIdx + half_idx)
            
            # for dataIdx in range(len(pelvis_x[trialIdx])):
    
            #     if diffVal[dataIdx] < -0.01:    #-50:
            #         tempMoveIdx.append(dataIdx)
            #     elif diffVal[dataIdx] > 0.01 and : # 50:
            #         tempSitIdx.append(dataIdx)

            tempMoveIdx = np.array(tempMoveIdx)
            tempSitIdx = np.array(tempSitIdx)

            temp_startMove_idx = np.min(tempMoveIdx)
            temp_endSitIdx = np.max(tempSitIdx)

            starMoveIdx.append(temp_startMove_idx)
            moveEndIdx.append(temp_endSitIdx)
            

            if displayResult:
                print("start Move , end Sit ", np.min(
                    temp_startMove_idx), np.max(temp_endSitIdx))
                resultVal = np.array(
                    [starMoveIdx[trialIdx], moveEndIdx[trialIdx]])
                curlTime_sec = np.array(
                    [self.timestamp_sec[trialIdx][temp_startMove_idx], self.timestamp_sec[trialIdx][temp_endSitIdx]])
                curlPelvis = np.array(
                    [pelvis_y[trialIdx][temp_startMove_idx], pelvis_y[trialIdx][temp_endSitIdx]])

                fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))
                ax[0].plot(self.timestamp_sec[trialIdx], pelvis_x[trialIdx],
                           'b-', label='filtered pelvis_X')
                title ="moveStartIdx: "+ str(temp_startMove_idx) + " , MoveEnd" + str(temp_endSitIdx)
                ax[0].set_title(" filtered pelvis X axis " + title)

                ax[0].axvline(x=self.timestamp_sec[trialIdx][temp_startMove_idx],
                              color='r', linestyle="--", linewidth=3, label='start Move')
                ax[0].axvline(x=self.timestamp_sec[trialIdx][temp_endSitIdx],
                              color='r', linestyle=":", linewidth=3, label='end Sit')

                ax[1].plot(self.timestamp_sec[trialIdx], pelvis_y[trialIdx], 'r-',
                           linewidth=1, label='filtered pelvis_y')
                ax[1].set_title(" filtered pelvis Y axis " + title)
                ax[1].axvline(x=self.timestamp_sec[trialIdx][temp_startMove_idx],
                              color='r', linestyle="--", linewidth=3)
                ax[1].axvline(x=self.timestamp_sec[trialIdx][temp_endSitIdx],
                              color='r', linestyle=":", linewidth=3)

                ax[2].plot(self.timestamp_sec[trialIdx], pelvis_z[trialIdx], 'g-',
                           linewidth=1, label='filtered pelvis_z')
                ax[2].set_title(" filtered pelvis Z axis " + title)
                ax[2].set_xlabel('Time [Sec]')
                ax[2].set_ylabel(
                    'Vertical Axis of Pelvis [mm]')
                ax[2].axvline(x=self.timestamp_sec[trialIdx][temp_startMove_idx],
                              color='r', linestyle="--", linewidth=3)
                ax[2].axvline(x=self.timestamp_sec[trialIdx][temp_endSitIdx],
                              color='r', linestyle=":", linewidth=3)
                fig.legend()
                plt.show()
        return starMoveIdx, moveEndIdx


    def getOneHot_actionLabel(self, actionIdx, label_code, numActions, displayResult=False):
        oneHot_actionList = []
        for trialIdx in range(len(self.eachTrialDir)): 
            actionList = []
            # actionIdx = # moveStartIdx ('0), startWalkIdx (1), turnStartIdx (2), turnEndIdx (3), startSitIdx (4), moveEndIdx (5)
            #  label_code =  "sitting": 0, "sit-stand":1, "walking":2, "turning":3, "stand-sit":4
            for timeIdx in range(len(self.timestamp_sec[trialIdx])):
                # moveStartIdx 전까지 = sitting
                if (timeIdx < actionIdx[0][trialIdx]):
                    actionList.append([1, 0, 0, 0, 0])
                # moveStartIdx ~ startWalk 까지 (sit-stand)
                elif (timeIdx >= actionIdx[0][trialIdx]) and (timeIdx < actionIdx[1][trialIdx]):
                    actionList.append([0, 1, 0, 0, 0])
                # startWalk ~ turnStartIdx 까지 (walking)
                elif (timeIdx >= actionIdx[1][trialIdx]) and (timeIdx < actionIdx[2][trialIdx]):
                    actionList.append([0, 0, 1, 0, 0])
                # turnStartIdx ~ turnEndIdx 까지 (turning)
                elif (timeIdx >= actionIdx[2][trialIdx]) and (timeIdx < actionIdx[3][trialIdx]):
                    actionList.append([0, 0, 0, 1, 0])
                # turnEndIdx ~ startSitIdx 까지 (walking)
                elif (timeIdx >= actionIdx[3][trialIdx]) and (timeIdx < actionIdx[4][trialIdx]):
                    actionList.append([0, 0, 1, 0, 0])
                # startSitIdx ~ moveEndIdx 까지 (stand-sit)
                elif (timeIdx >= actionIdx[4][trialIdx]) and (timeIdx < actionIdx[5][trialIdx]):
                    actionList.append([0, 0, 0, 0, 1])
                # startSitIdx ~ moveEndIdx 까지 (sitting)
                elif (timeIdx >= actionIdx[5][trialIdx]):
                    actionList.append([1, 0, 0, 0, 0])
                else:
                    AssertionError(
                        "Check getLabel_forEveryTrial func. in utils_labeling_all")

            # (num or data for one trial, 1)
            oneTrial_actionList = np.array(actionList)
            oneHot_actionList.append(oneTrial_actionList)
        if displayResult:
            print("One trial shape: ", np.array(oneTrial_actionList).shape)

        return np.array(oneHot_actionList)


    def bindData_oneHotLabel(self, actionList, numFeature, numActions, displayResult=False):
        labelData_AllTrial = []
        pelvis_x = self.rawPelvis_x
        pelvis_y = self.rawPelvis_y
        pelvis_z = self.rawPelvis_z
        for trialIdx in range(len(self.eachTrialDir)):
            # 4 (time,x,y,z) + 5 (# "sitting": 0, "sit-stand":1, "walking":2, "turning":3, "stand-sit":4)
            dataset_columns = numFeature + numActions
            rawData = np.array([self.timestamp_sec[trialIdx], pelvis_x[trialIdx],
                                pelvis_y[trialIdx], pelvis_z[trialIdx]])  # (4,317)
            unLabel_rawData = rawData.T   # (317,4)
            # print(rawData.shape, unLabel_rawData.shape)
            # unLabel_rawData = np.array([timestamp_sec, pelvis_y])

            # print(actionList[trialIdx].shape )
            # (eachTrial_NumOfData, 6)
            labelData = np.zeros((len(unLabel_rawData), dataset_columns))
            # time , pelvis  넣음  (317,4)
            labelData[:, :-(numActions)] = unLabel_rawData
            # (335, 5) one hot encoding 수행한 actionsList
            labelData[:,  -(numActions):] = actionList[trialIdx]

            labelData_AllTrial.append(labelData)
        if displayResult:
            # (trialIdx, 8)  # (trialIdx, 1, 8)
            print("Final Data shape:  ", np.array(labelData_AllTrial).shape)

        return np.array(labelData_AllTrial)


    def saveLabeledData(self, ROOT_DIR, expDateFolder, labelData, displayResult=False, saveOption='Labeled_CSV'):
        savePathRoot = "/home/sang/dataset/TUG/arrangedData_LabelSave"
        os.chdir(savePathRoot)
        for trialIdx in range(len(self.eachTrialDir)):
            saveDatePath = savePathRoot + "/" + expDateFolder
            if os.path.isdir(saveDatePath) is False:  # 폴더 없는 경우
                os.mkdir(saveDatePath)

            saveOptionPath = saveDatePath + "/" + saveOption
            if os.path.isdir(saveOptionPath) is False:  # 폴더 없는 경우
                os.mkdir(saveOptionPath)

            saveSubPath = saveOptionPath + "/" + self.subjectFolderlist[trialIdx]
            if os.path.isdir(saveSubPath) is False:  # 폴더 없는 경우
                os.mkdir(saveSubPath)

            filePath = saveSubPath + "/"
            fileName = filePath + 'Labeled_CSV_' + \
                self.subjectFolderlist[trialIdx] + "_" + \
                self.trialFolderlist[trialIdx]+".csv"

            if displayResult:
                print("[save CSV] ", fileName)
            np.savetxt(fileName, labelData[trialIdx], fmt='%s', delimiter=',')

        os.chdir(ROOT_DIR)