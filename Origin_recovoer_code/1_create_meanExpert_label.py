import pandas as pd
import os,re

def get_frame_number(name):
    if not name:
        return 0
    x = [int(s) for s in name[:-4].split('_') if s.isdigit()]
    return x[0]
    

folder1 = 'LSTM2HMM_saveResults_illness_KYH'
folder2 = 'LSTM2HMM_saveResults_illness_KHJ'

dataset_type = 'norm60'

path = 'C:\my_stuff\yj\{folder}\\0_sideView\\'+dataset_type


dir_tuples = [tuple for tuple in os.walk(path.format(folder=folder1)) if not tuple[1] and (any('label') in file for file in tuple[2])]

#print(dir_tuples)

li = []
for tuple in dir_tuples:
    seperated_path = tuple[0].split('\\')
    subject = seperated_path[-2]
    trial = seperated_path[-1]
    csvname = 'label_'+subject+'_'+trial+'.csv'
    try:
        path_kyh = os.path.join(tuple[0], csvname)
        df = pd.read_csv(path_kyh)
        path_khj = path_kyh.replace(folder1, folder2)
        df_from_each_file = (pd.read_csv(f) for f in (path_khj,path_kyh))
        concatenated_df   = pd.concat(df_from_each_file, ignore_index=True).applymap(get_frame_number)
        concatenated_df['subject'] = subject
        concatenated_df['trial'] = trial
        li.append(concatenated_df)
    except:
        print("Except File List (Subject, Trial): ", subject,"_", trial)
        continue

#print(get_frame_number('color_00299.jpg'))
pd.concat(li, axis=0, ignore_index=True).groupby(['subject','trial'], as_index=False).mean().round(0).sort_values(by=['subject','trial']).to_csv(
    'C:\my_stuff\yj\\mean_{dataset_type}.csv'.format(dataset_type=dataset_type),index=False)

