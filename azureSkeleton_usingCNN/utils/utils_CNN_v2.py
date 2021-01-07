import sys
import os
import copy
import fnmatch
import matplotlib.pyplot as plt
import numpy as np


class utilsCNN:
    def __init__(self):
        super().__init__()  # overriding 금지 (상속 안해서 상관 없긴 한데..)
        print("utils CNN 객체 생성 -- Initialize")
        self.eachCSV_Dir = []  # subjectFoloderlist
        self.dateFolderlist = []
        self.subjectFolderlist = []
        self.csvFilelist = []
        self.totalSubjectNum =0
        self.totalCSVfiles = 0

        self.timestamp_sec = []
        self.rawPelvis_x = []
        self.rawPelvis_y = []
        self.rawPelvis_z = []
        self.oneHot_action = []

        self.sitting = []
        self.stand_sit = []
        self.walking = []
        self.turning = []
        self.stand_sit = []

    def getDatalist_forEveryTrials(self, ROOT_DIR, dataForm="Labeled_CSV", displayList=False):
        # ---- check every data in Root Folder

        for dateIndex, dateFolder in enumerate(sorted(os.listdir(ROOT_DIR))):
            if displayList:
                print("Folder index: {0} ,    Folder name: {1} ".format(
                    dateIndex, dateFolder))
            dateFolder_DIR = ROOT_DIR+"/"+dateFolder + "/" + dataForm
            for subjectIndex, subjectName in enumerate(sorted(os.listdir(dateFolder_DIR))):
                subjectFolder_DIR = dateFolder_DIR + "/" + subjectName
                if displayList:
                    print(
                        "     ----------------------------------------------------------------------         ")
                    self.eachCSV_Dir.append(subjectFolder_DIR)
                    self.dateFolderlist.append(dateFolder)
                    self.subjectFolderlist.append(subjectName)
                    

                for (dir_path, dir_folder, files) in os.walk(subjectFolder_DIR):
                    csvlist = []
                    if len(files) > 0:
                        for fileName in sorted(files):
                            if fnmatch.fnmatch(fileName, "*.csv"):
                                csvlist.append(fileName)
                                # self.csvFilelist.append(fileName)
                                self.totalCSVfiles += 1  # of
                    # self.csvFilelist.append(csvlist) # [Subject folder 수 ][csv file 수]
                # [Subject folder 수 ][csv file 수]
                self.csvFilelist.append(csvlist)

                if displayList:
                    print("     Subject Idx: {0},      Subject: {1} ,   # fo csvFiles: {2}".format(
                        self.totalSubjectNum, subjectName,  len(self.csvFilelist[subjectIndex])))
                self.totalSubjectNum+=1
        os.chdir(ROOT_DIR)
        print("-----> [ Total Trials] (# of Trial): ", self.totalCSVfiles)

    def getData_forEveryTrial(self, ROOT_DIR, displayResult=False):
        self.timestamp_sec = []
        self.rawPelvis_x = []
        self.rawPelvis_y = []
        self.rawPelvis_z = []
        self.oneHot_action = []

        # subjectName 폴더 안에 csv 모여있음
        for subjectIndex in range(len(self.eachCSV_Dir)):
            os.chdir(self.eachCSV_Dir[subjectIndex])
            os.getcwd()
            # csv file 읽어오기
            tempFrame = []
            tempPelvis_x = []
            tempPelvis_y = []
            tempPelvis_z = []
            temp_action = []
            for csvIdx in range(len(self.csvFilelist[subjectIndex])):
                rawData = np.loadtxt(
                    self.csvFilelist[subjectIndex][csvIdx], delimiter=",")

                tempFrame.append(rawData[:, 0])
                tempPelvis_x.append(rawData[:, 1])
                tempPelvis_y.append(rawData[:, 2])
                tempPelvis_z.append(rawData[:, 3])
                # {"sitting": 0, "sit-stand":1, "walking":2, "turning":3, "stand-sit":4}
                temp_action.append(rawData[:, 4: 9])

            self.timestamp_sec.append(tempFrame)
            self.rawPelvis_x.append(tempPelvis_x)
            self.rawPelvis_y.append(tempPelvis_y)
            self.rawPelvis_z.append(tempPelvis_z)
            self.oneHot_action.append(temp_action)

            csvfileIdx = 0
            if displayResult:
                print(np.array(self.timestamp_sec[subjectIndex]).shape,
                      np.array(self.rawPelvis_x[subjectIndex]).shape,
                      np.array(self.oneHot_action[subjectIndex][csvfileIdx]).shape)

            os.chdir(ROOT_DIR)
        else:
            AssertionError("image Index is not in a range")

    def createDataset_fromCSV(self, ROOT_DIR, numFeature, numActions, trainSubNum, displayInfo=False):
        train_x = np.zeros((0, numFeature))
        train_y = np.zeros((0, numActions))
        test_x = np.zeros((0, numFeature))
        test_y = np.zeros((0, numActions))

        for subjectIndex in range(len(self.eachCSV_Dir)):
            os.chdir(self.eachCSV_Dir[subjectIndex])
            for trialIdx in range(len(self.csvFilelist[subjectIndex])):

                rawData = np.array([self.timestamp_sec[subjectIndex][trialIdx], self.rawPelvis_x[subjectIndex][trialIdx],
                                    self.rawPelvis_y[subjectIndex][trialIdx], self.rawPelvis_z[subjectIndex][trialIdx]])  # (4, 9)
                unLabel_rawData = rawData.T   # (9,4)

                if subjectIndex < trainSubNum:
                    if displayInfo:
                        print(
                            "Dir: ", self.eachCSV_Dir[subjectIndex], " ,   FileName: ", self.csvFilelist[subjectIndex][trialIdx])
                    train_x = np.append(train_x, unLabel_rawData, axis=0)
                    train_y = np.append(
                        train_y, self.oneHot_action[subjectIndex][trialIdx], axis=0)
                else:
                    if displayInfo:
                        print(
                            "Dir: ", self.eachCSV_Dir[subjectIndex], " ,   FileName: ", self.csvFilelist[subjectIndex][trialIdx])
                    test_x = np.append(test_x, unLabel_rawData, axis=0)
                    test_y = np.append(
                        test_y, self.oneHot_action[subjectIndex][trialIdx], axis=0)

        os.chdir(ROOT_DIR)
        return train_x, train_y, test_x, test_y

    def createDatasetSplit_fromCSV(self, ROOT_DIR, numFeature, numActions, trainSubNum, displayInfo=False):
        dataset_columns = numFeature + numActions  # 4+ 5
        train_x = np.zeros((0, numFeature))
        train_y = np.zeros((0, numActions))
        test_x = np.zeros((0, numFeature))
        test_y = np.zeros((0, numActions))

        for subjectIndex in range(len(self.eachCSV_Dir)):
            os.chdir(self.eachCSV_Dir[subjectIndex])
            for trialIdx in range(len(self.csvFilelist[subjectIndex])):

                rawData = np.loadtxt(
                    self.csvFilelist[subjectIndex][trialIdx], delimiter=",")
                dataset_columns = numFeature + numActions
                rawData = np.array([self.timestamp_sec[subjectIndex][trialIdx], self.rawPelvis_x[subjectIndex][trialIdx],
                                    self.rawPelvis_y[subjectIndex][trialIdx], self.rawPelvis_z[subjectIndex][trialIdx]])  # (4, 9)
                unLabel_rawData = rawData.T   # (9,4)
                # labelData = np.zeros((len(unLabel_rawData), numFeature))
                # labelData[:, :-(numActions)] = unLabel_rawData
                # labelData[:,  -(numActions):] =  self.oneHot_action[subjectIndex][trialIdx]

                # if trialIdx < 8:
                #     train_data = np.append(test_data, labelData, axis = 0)
                # else:
                #     test_data = np.append(test_data, labelData, axis = 0)
                if subjectIndex < trainSubNum:
                    if displayInfo:
                        print(
                            "Dir: ", self.eachCSV_Dir[subjectIndex], " ,   FileName: ", self.csvFilelist[subjectIndex][trialIdx])
                    # train_data = np.append(train_data, labelData, axis = 0)
                    train_x = np.append(train_x, unLabel_rawData, axis=0)
                    train_y = np.append(
                        train_y,  self.oneHot_action[subjectIndex][trialIdx], axis=0)
                else:
                    if displayInfo:
                        print(
                            "Dir: ", self.eachCSV_Dir[subjectIndex], " ,   FileName: ", self.csvFilelist[subjectIndex][trialIdx])
                    test_x = np.append(test_x, unLabel_rawData, axis=0)
                    test_y = np.append(
                        test_y,  self.oneHot_action[subjectIndex][trialIdx], axis=0)

        os.chdir(ROOT_DIR)
        return train_x, train_y, test_x, test_y

    def time_series_2d_to_3d_section(self, data_x, data_y, numActions, sliding_window_size, step_size_of_sliding_window, standardize=False, **options):
        data = data_x  # sensor data
        act_labels = data_y  # action labels
        mean = 0
        std = 1

        if standardize:
            # As usual, normalize test dataset by training dataset's parameters
            if options:
                mean = options.get("mean")
                std = options.get("std")
                # \n test mean: {0}, std: {1}\n\n".format(mean,std))
                print("Test Data has been standardized:")
            else:
                # csv 한 줄의 mean ..이걸 모든 sensor data들에 대해 수행
                mean = data.mean(axis=0)
                std = data.std(axis=0)
                print("Training Data has been standardized:\n the mean is = ", str(
                    mean.mean()), " ; and the std is = ", str(std.mean()))
            # mean removal and variance scaling
            data -= mean
            data /= std
        else:
            print("----> Without Standardization.....")

        # We want the Rows of matrices show each Feature and the Columns show time points.
        # before transepose,,(145687, 12) -->  After (12, 145687)
        data = data.T

        size_features = data.shape[0]  # 12
        size_data = data.shape[1]

        number_of_secs = round(
            ((size_data - sliding_window_size)/step_size_of_sliding_window))

        # Create a 3D matrix for Storing Snapshots
        secs_data = np.zeros(
            (number_of_secs, size_features, sliding_window_size))
        act_secs_labels = np.zeros((number_of_secs, numActions))
        print("number of sections: ", number_of_secs)
        k = 0
        for i in range(0, (size_data) - sliding_window_size, step_size_of_sliding_window):
            j = i // step_size_of_sliding_window
            if(j >= number_of_secs):
                break

            if(not (act_labels[i] == act_labels[i+sliding_window_size-1]).all()):
                continue
            secs_data[k] = data[0:size_features, i:i+sliding_window_size]
            act_secs_labels[k] = act_labels[i].astype(int)
            k = k+1
        print( secs_data[0][0])
        secs_data = secs_data[0:k]    
        act_secs_labels = act_secs_labels[0:k]

        return secs_data, act_secs_labels, mean, std
