import os
import numpy as np

from scipy.signal import butter, lfilter, filtfilt  
from scipy.interpolate import UnivariateSpline  # missing value -> np.nan -> to zero -> 
from sklearn.preprocessing import MinMaxScaler


class preprocess:
    def __init__(self, datasetDir, expertFolder):
        self.datasetDir = datasetDir
        self.expertFolder = expertFolder
        
        self.continuous_errorVal_Check = 20
        self.lookback_window = 8
        
        self.numFeature = 4 # time + numPelvis = 4
        self.numPelvis = 3 
        self.numActions = 5
        self.dataset_columns = self.numFeature + self.numActions
        
    def get_subjectList(self, shuffle = False):      
        expDates = next(os.walk(self.datasetDir))[1]   #['2020_11_03', ...] 
        subject_number = 0
        subjects_list = []

        for dateFolder in expDates:
            dateDir = os.path.join(self.datasetDir, dateFolder)   # ..../trainSet/..saveResults_최윤정/0_sideView\2020_11_03
            tmpSubjects = next(os.walk(dateDir))[1]
            subjects_list.append(tmpSubjects)  

        # ------ 2D -> 1D [[sub1, sub2], [sub3, sub4]] -> [sub1, sub2, sub3, sub4]
        subjects = []
        for eachSub in subjects_list:
            for sub in eachSub:
                if ".ipynb_checkpoints" == sub:
                    pass
                else:
                    subjects.append(sub)
                    subject_number +=1

        if shuffle:
            np.random.shuffle(subjects)

        print(" 1D sub list: " , subjects, len(subjects))
        return subjects, subject_number  
    
    # ------------------------Read Raw or Norm data -----------------
    def get_rawData(self, csvFile):
        interCheck = False
      
        try:
            csv_data_ = np.loadtxt(csvFile, delimiter=',', dtype=str, skiprows=2, usecols=(0,1,2,3)) 
            frameNum = csv_data_[:,0].astype(np.int)  # int로 type conversion
            max_frame = len(frameNum)

            rawPelvis_x = np.array(frameNum).astype(np.float32)
            rawPelvis_y = np.array(frameNum).astype(np.float32)
            rawPelvis_z = np.array(frameNum).astype(np.float32)


            #--- if have "empty value" ---- Interpolation  
            for i, v in enumerate(csv_data_[:, 1]):
                if v == " ": 
                    interCheck = True
                    csv_data_[i,1] = np.nan
                    csv_data_[i,2] = np.nan
                    csv_data_[i,3] = np.nan
                rawPelvis_x[i] = csv_data_[i, 1].astype(np.float32)
                rawPelvis_y[i] = csv_data_[i, 2].astype(np.float32)
                rawPelvis_z[i] = csv_data_[i, 3].astype(np.float32)

            if interCheck: 
                w_x = np.isnan(rawPelvis_x) 
                rawPelvis_x[w_x] = 0
                Fx = UnivariateSpline(frameNum,rawPelvis_x,w=~w_x)
                newPelvis_x = Fx(frameNum)

                w_y = np.isnan(rawPelvis_y) 
                rawPelvis_y[w_y] = 0
                Fy = UnivariateSpline(frameNum,rawPelvis_y,w=~w_y)
                newPelvis_y = Fy(frameNum)

                w_z = np.isnan(rawPelvis_z) 
                rawPelvis_z[w_z] = 0
                Fz = UnivariateSpline(frameNum,rawPelvis_z,w=~w_z)
                newPelvis_z = Fz(frameNum)

            else:
                newPelvis_x = rawPelvis_x
                newPelvis_y = rawPelvis_y
                newPelvis_z = rawPelvis_z


            if np.max(newPelvis_z) >= 5000:# ----------- if have garbage value (not "empty..") ...        
                for i in range(self.continuous_errorVal_Check): # 30 
                    newPelvis_x[np.where(np.abs(newPelvis_x[1+i:]-newPelvis_x[:-(1+i)]) > 100)[0]] = newPelvis_x[np.where(np.abs(newPelvis_x[1+i:]-newPelvis_x[:-(1+i)]) > 100)[0]-1]
                    newPelvis_y[np.where(np.abs(newPelvis_y[1+i:]-newPelvis_y[:-(1+i)]) > 100)[0]] = newPelvis_y[np.where(np.abs(newPelvis_y[1+i:]-newPelvis_y[:-(1+i)]) > 100)[0]-1]
                    newPelvis_z[np.where(np.abs(newPelvis_z[1+i:]-newPelvis_z[:-(1+i)]) > 100)[0]] = newPelvis_z[np.where(np.abs(newPelvis_z[1+i:]-newPelvis_z[:-(1+i)]) > 100)[0]-1]


            timeData = np.loadtxt(csvFile, delimiter=',', skiprows=2, dtype=np.float32, usecols=[97, 98, 99, 100]) #  # skiprows=2 (header + first data.. 가끔 첫 줄 data 이상함)
            timestamp_diff_msec = np.array([rawPelvis_x])
            for i in range(max_frame):
                before_time_msec = timeData[i-1, 0] * 60 * 60 * 1000 + timeData[i-1, 1] * 60 * 1000 +  timeData[i-1, 2] * 1000 + timeData[i-1, 3] 
                current_time_msec = timeData[i, 0] * 60 * 60 * 1000 + timeData[i, 1] * 60 * 1000 + timeData[i, 2] * 1000 + timeData[i, 3]
                if i is 0:
                    timestamp_diff_msec[0, i] = 0
                else:
                    timestamp_diff_msec[0, i] = int(current_time_msec - before_time_msec)

                time_sum_msec = 0
                eachTimestamp_sec = np.array(frameNum, dtype=float) # sec (?,1)
                for timeIdx in range(timestamp_diff_msec.shape[1]):
                    time_sum_msec = int(time_sum_msec + timestamp_diff_msec[0, timeIdx])
                    eachTimestamp_sec[timeIdx] = time_sum_msec / 1000  

            return frameNum, eachTimestamp_sec, newPelvis_x, newPelvis_y, newPelvis_z
        except:
            pass

    
    
    def getLPF_PelvisData(self, csvFile):
        rawData = np.loadtxt(csvFile, delimiter=",")
        timestamp = rawData[:,0]
        pelvis_x = rawData[:,1]
        pelvis_y = rawData[:,2]
        pelvis_z = rawData[:,3]
        actionList = rawData[:,4:]
        return timestamp, pelvis_x, pelvis_y, pelvis_z, actionList
    

    
     # ------------------------ Pre-Processing Pelvis Data (LPF & Norm) -----------------
    def get_normData(self, pelvis_x, pelvis_y, pelvis_z, normMethod = 'min_max'):    
        if normMethod == "min_max":
            normPelvis_x = (pelvis_x -np.min(pelvis_x)) / ( np.max(pelvis_x) - np.min(pelvis_x))
            normPelvis_y = (pelvis_y -np.min(pelvis_y)) / ( np.max(pelvis_y) - np.min(pelvis_y))
            normPelvis_z = (pelvis_z -np.min(pelvis_z)) / ( np.max(pelvis_z) - np.min(pelvis_z))

        elif normMethod == "z_score":
            normPelvis_x = (pelvis_x -np.mean(pelvis_x)) / np.std(pelvis_x)
            normPelvis_y = (pelvis_y -np.mean(pelvis_y)) / np.std(pelvis_y)
            normPelvis_z = (pelvis_z -np.mean(pelvis_z)) / np.std(pelvis_z)

        return normPelvis_x, normPelvis_y, normPelvis_z

    def get_LPFData(self, pelvis_x, pelvis_y, pelvis_z, cutoff_freq = 0.6, order=2):
        sampling_freq = 30
        number_of_samples = len(pelvis_y)
        normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
        numerator_coeffs, denominator_coeffs = butter(order, normalized_cutoff_freq)
        Filtered_x = filtfilt(numerator_coeffs, denominator_coeffs, pelvis_x)  # Phase shift 없음
        Filtered_y = filtfilt(numerator_coeffs, denominator_coeffs, pelvis_y)
        Filtered_z = filtfilt(numerator_coeffs, denominator_coeffs, pelvis_z)
        return Filtered_x, Filtered_y, Filtered_z
   
    # --------------------------------------------- Action Label Related -----------------
    def get_actFrameNumber(self,csvFile): 
        actFrame = []
        with open(csvFile, 'r', encoding='utf-8') as readCSV:
            data = readCSV.read()
            lines = data.split("\n") # lines[0] = header, [1] = act label data
            tmpLabel = lines[1].split(",")

            for i in range(len(tmpLabel)):           
                actFrame.append( int(tmpLabel[i].split("_")[1].split(".")[0]) ) 
            return actFrame   

    #  Origin... sit, sit-to-stand, walk, turn, stand-to-sit (5action)
    def convert_act2oneHot(self, actFrame, frameNum):
        actionList = []
        for timeIdx in range(len(frameNum)):
            if (timeIdx < actFrame[0]):  # frame Num < move start fram (sit)
                actionList.append(np.array([1, 0, 0, 0, 0],dtype=np.int))
                        
            elif (timeIdx >= actFrame[0]) and (timeIdx < actFrame[1]):  # moveStartIdx ~ startWalk 까지 (sit-to-stand)
                actionList.append(np.array([0, 1, 0, 0, 0],dtype=np.int))
                        
            elif (timeIdx >= actFrame[1]) and (timeIdx < actFrame[2]):  # startWalk ~ turnStartIdx 까지 (walk)
                actionList.append(np.array([0, 0, 1, 0, 0],dtype=np.int))
                        
            elif (timeIdx >= actFrame[2]) and (timeIdx < actFrame[3]):  # turnStartIdx ~ turnEndIdx 까지 (turn)
                actionList.append(np.array([0, 0, 0, 1, 0],dtype=np.int))
                       
            elif (timeIdx >= actFrame[3]) and (timeIdx < actFrame[4]): # turnEndIdx ~ startSitIdx 까지 (walk)
                actionList.append(np.array([0, 0, 1, 0, 0],dtype=np.int))
                      
            elif (timeIdx >= actFrame[4]) and (timeIdx < actFrame[5]):  # startSitIdx ~ moveEndIdx 까지 (stand-to-sit)
                actionList.append(np.array([0, 0, 0, 0, 1],dtype=np.int))

            elif (timeIdx >= actFrame[5]):
                actionList.append(np.array([1, 0, 0, 0, 0],dtype=np.int))


            else:
                AssertionError("Check getLabel_forEveryTrial func. in utils_labeling_all")
            oneTrial_actionList = np.array(actionList)
        return np.array(oneTrial_actionList)

    
    
    
    
    # ---------------------- Create Dataset (Train / Validation / Test ) --------------------- #
    def sliding_window(self, timestamp, data_x, data_y):
        time = []
        x = []
        y = []
#         enc = MinMaxScaler(feature_range = (0, 1))
#         enc_y = enc.fit_transform(data_y)
        for i in range(self.lookback_window, len(data_x)+1):
            time.append(timestamp[i-1])
            x.append(data_x[i - self.lookback_window:i])
            y.append(data_y[i-1])

        x = np.array(x)
        x = x.reshape(-1, self.lookback_window, self.numPelvis, 1)

        y = np.array(y)
        time = np.array(time)

#         return time, x, y, enc
        return time, x, y
    
    
    def readLPF_createDataset(self, specific_group, train_subjects, test_subjects, Foldername="Originact5_lpf_", Kfold=False):
        train_x = np.zeros((0, self.lookback_window, self.numPelvis , 1)) 
        train_y = np.zeros((0, self.numActions))

        valid_x = np.zeros((0, self.lookback_window, self.numPelvis, 1))
        valid_y = np.zeros((0, self.numActions))

        test_x = np.zeros((0, self.lookback_window, self.numPelvis,1 ))
        test_y = np.zeros((0, self.numActions)) 

        total_Trials = 0
        for dirpath, foldername, files in sorted(os.walk(self.datasetDir)):
            actPath = None
            pelvisPath= None
            actFrame = []
            if ".ipynb_checkpoints" in dirpath:
                continue
            else:
                for filename in sorted(files):
                    if ".csv" in filename:
                        if Foldername in filename:     # 
                            pelvisPath = dirpath    
                            subname = dirpath.split("/")[-2] 
                            pelvis_csv = os.path.join(dirpath, filename)

                            timstamp_ms, lpfPelvis_x, lpfPelvis_y, lpfPelvis_z, oneHot_actionList  = self.getLPF_PelvisData(pelvis_csv)  

                            total_Trials +=1   

                            pelvisData = np.array([lpfPelvis_x, lpfPelvis_y, lpfPelvis_z]).T  # (335,3) 
                            actionData = np.array(oneHot_actionList)    # # (335, 5)

                            blockTime, blockPelvis, blockLable = self.sliding_window(timstamp_ms, data_x = pelvisData, data_y = actionData) # pelvis - (327, 8, 3)
    #                         print(dirpath, "\t", blockLable.shape)
                            if Kfold:                                       
                                if subname in train_subjects:
                                    train_x = np.append(train_x, blockPelvis, axis = 0 )
                                    train_y = np.append(train_y, blockLable, axis = 0 )
                                elif subname in test_subjects:
                                    test_x = np.append(test_x, blockPelvis, axis = 0 )
                                    test_y = np.append(test_y, blockLable, axis = 0 )  
                            else:      
                                if subname in (specific_group and train_subjects):
                                    train_x = np.append(train_x, blockPelvis, axis = 0 )
                                    train_y = np.append(train_y, blockLable, axis = 0 )

                                elif subname in (specific_group and test_subjects):
                                    test_x = np.append(test_x, blockPelvis, axis = 0 )
                                    test_y = np.append(test_y, blockLable, axis = 0 ) 
                                elif subname in specific_group:
                            
                                    valid_x = np.append(valid_x, blockPelvis, axis = 0 )
                                    valid_y = np.append(valid_y, blockLable, axis = 0 )
                                                                      
        return total_Trials, train_x, train_y, valid_x, valid_y, test_x, test_y




    def readRaw_createDataset(self, specific_group, train_subjects, test_subjects, Kfold):
        train_x = np.zeros((0, self.lookback_window, self.numPelvis , 1)) 
        train_y = np.zeros((0, self.numActions))

        valid_x = np.zeros((0, self.lookback_window, self.numPelvis, 1))
        valid_y = np.zeros((0, self.numActions))

        test_x = np.zeros((0, self.lookback_window, self.numPelvis,1 ))
        test_y = np.zeros((0, self.numActions)) 

        total_Trials = 0
        for dirpath, foldername, files in sorted(os.walk(self.datasetDir)):
            clear_output(wait=True)
            actPath = None
            pelvisPath= None
            actFrame = []
            if ".ipynb_checkpoints" in dirpath:
                continue
            else:
                for filename in sorted(files):
                    if ".csv" in filename:
                        if "label_" in filename:
                            labelPath = dirpath
                            label_csv = os.path.join(labelPath,filename)
                            actFrame = get_actFrameNumber( label_csv)

                        if "skeleton_" in filename:          
                            pelvisPath = dirpath    
                            subname = dirpath.split("/")[-2] 
                            pelvis_csv = os.path.join(dirpath, filename)

                            frameNum, timstamp_ms, pelvis_x, pelvis_y, pelvis_z  = self.get_rawData(pelvis_csv)  # raw                 
                            normPelvis_x, normPelvis_y, normPelvis_z = self.get_normData(pelvis_x, pelvis_y, pelvis_z, normMethod = 'min_max') # norm         
                            lpfPelvis_x, lpfPelvis_y, lpfPelvis_z = self.get_LPFData(normPelvis_x, normPelvis_y, normPelvis_z, cutoff_freq=0.5, order=1)  # lpf 

                            total_Trials +=1   

                            if labelPath == pelvisPath:  
                                oneHot_actionList = self.convert_act2oneHot(actFrame, frameNum)

                                pelvisData = np.array([lpfPelvis_x, lpfPelvis_y, lpfPelvis_z]).T  # (335,3) 
                                actionData = np.array(oneHot_actionList)    # # (335, 5)

                                blockTime, blockPelvis, blockLable = self.sliding_window(timstamp_ms, data_x = pelvisData, data_y = actionData)

                            if Kfold:                                       
                                if subname in train_subjects:
                                    train_x = np.append(train_x, blockPelvis, axis = 0 )
                                    train_y = np.append(train_y, blockLable, axis = 0 )
                                elif subname in test_subjects:
                                    test_x = np.append(test_x, blockPelvis, axis = 0 )
                                    test_y = np.append(test_y, blockLable, axis = 0 )  
                            else:      
                                if subname in (specific_group and train_subjects):
                                    train_x = np.append(train_x, blockPelvis, axis = 0 )
                                    train_y = np.append(train_y, blockLable, axis = 0 )

                                elif subname in (specific_group and test_subjects):
                                    test_x = np.append(test_x, blockPelvis, axis = 0 )
                                    test_y = np.append(test_y, blockLable, axis = 0 ) 
                                elif subname in specific_group:
                                    valid_x = np.append(valid_x, blockPelvis, axis = 0 )
                                    valid_y = np.append(valid_y, blockLable, axis = 0 )

        return total_Trials, train_x, train_y, valid_x, valid_y, test_x, test_y