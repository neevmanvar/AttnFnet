 #%% Prepare Dataset
import os
import glob
import csv
import sys
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

def get_filtered_pm(Arr, pm_adjust_cm, weights, save_mmhg = False, use_filter=False, NORM_INDIVIDUAL=False):
    if Arr.ndim !=3:
        raise ValueError("array dimensions must be 3")
    
    if Arr.shape[1] < Arr.shape[2]:
        raise ValueError("array must be portrait meaning 190,84 dimensions")

    scaling_factors = []
    filtered_arr = []
    for PM_arr in Arr:
        if pm_adjust_cm[1] <= -1:
            PM_arr = PM_arr[1-pm_adjust_cm[1]:192, 0:84]  # cut off the edges because the original pressure mat is like 1.90 x 0.77 while this one is 1.92 x 0.84.
            if np.shape(PM_arr)[0] < 190:
                PM_arr = np.concatenate((PM_arr, np.zeros((190-np.shape(PM_arr)[0], np.shape(PM_arr)[1]))), axis = 0)

        elif pm_adjust_cm[1] == 0: #this is if you have 0 through 192 or 1 through 191
            #print('got here')
            PM_arr = PM_arr[1:191, :]

        elif pm_adjust_cm[1] >= 1:
            PM_arr = PM_arr[0:191-pm_adjust_cm[1], 0:84]
            if np.shape(PM_arr)[0] < 190:
                PM_arr = np.concatenate((np.zeros((190-np.shape(PM_arr)[0], np.shape(PM_arr)[1])), PM_arr), axis = 0)


        if pm_adjust_cm[0] <= -4:
            PM_arr = PM_arr[:, 3-pm_adjust_cm[0]:84]
            if np.shape(PM_arr)[1] < 77:
                PM_arr = np.concatenate((PM_arr, np.zeros((np.shape(PM_arr)[0], 77-np.shape(PM_arr)[1]))), axis = 1)

        elif pm_adjust_cm[0] >= -3 and pm_adjust_cm[0] <= 2:
            #for a -2 you want it like [6:83]
            #for a 3 you want it like [1:78]
            PM_arr = PM_arr[:, 3-pm_adjust_cm[0]:80-pm_adjust_cm[0]]

        elif pm_adjust_cm[0] >= 3:
            PM_arr = PM_arr[:, 0:80-pm_adjust_cm[0]]
            if np.shape(PM_arr)[1] < 77:
                PM_arr = np.concatenate((np.zeros((np.shape(PM_arr)[0], 77-np.shape(PM_arr)[1])), PM_arr), axis = 1)
        
        # PM_arr = PM_arr[4:183,:]   # y_train[:, 3:80, 1:191] # if we use original zoom factor
        if use_filter:
            PM_arr = gaussian_filter(PM_arr, sigma = 0.5, mode='reflect') # original sigma = 0.5/0.345
        PM_arr = zoom(PM_arr, (0.3355, 0.355), order=1)  # 0.355 original for both   # chenge it to 0.336 for auto zoom and no corp from above line
        
        # print("before press conver", np.min(PM_arr), np.max(PM_arr))

        if save_mmhg:
            MMHG_FACT = (1 / 133.322)
            KPA_FACT = (1/1000)
            No_FACT = 1
            PM_arr = PM_arr * ((weights[i] * 9.81) / (np.sum(PM_arr) * 0.0264 * 0.0286)) * KPA_FACT

        if NORM_INDIVIDUAL:
            scaling_factors.append(np.max(PM_arr))
            PM_arr = PM_arr/np.max(PM_arr)

        filtered_arr.append(PM_arr)        
        # print("after press conver", np.min(PM_arr), np.max(PM_arr))
    return np.rot90(np.asarray(filtered_arr), k=3, axes=(1,2)), np.asarray(scaling_factors)


# use below instructions to remove npz files
x = input("Do you want to delete .npz files? (y/n): ")
npz_data_dir = "depth2bp_cleaned_no_KPa"
try:
   os.makedirs(npz_data_dir)
except:
   pass

if x.lower() == 'y':
   [os.remove(file) for file in glob.glob(npz_data_dir+'/*.npz')]
elif x.lower() == 'n':
   pass
else:
   print("input must be alphabatical (y/n)")
   os._exit(0)

USE_FILTER = True
SAVE_MMHG = True

dir_name = 'danaLab'
dir = dir_name + '/'
ids = os.listdir(dir_name)
ids = ids[:-11]

# get pandas Frame with path column to the image
import pandas as pd
total_img = 45
names_df = pd.DataFrame({'ids':ids,
                         'Depth_path': [[dir + str(i).zfill(5) + '/depthRaw/uncover/' + str(j).zfill(6) + '.npy' for j in range(1, 1+total_img)] for i in ids],
                         'Pressure_path': [[dir + str(i).zfill(5) + '/PMarray/uncover/' + str(j).zfill(6) + '.npy' for j in range(1, 1+total_img)] for i in ids],
                         'Pressure_cali_path': [dir + str(i).zfill(5)+"/PMcali.npy" for i in ids]
                         })

# set full length column display
pd.set_option('display.max_colwidth', None)
print(names_df.head())

#%% iniialize height and width
import numpy as np
depth_img = np.load(dir+'00001/depthRaw/uncover/000001.npy')
pressure_img = np.load(dir+'00001/PMarray/uncover/000001.npy')

depth_stream_height = depth_img.shape[0]
depth_stream_width = depth_img.shape[1]
pressure_stream_height = pressure_img.shape[0]
pressure_stream_width = pressure_img.shape[1]
print("depth height X width",depth_stream_height, depth_stream_width)
print("pressure height X width", pressure_stream_height, pressure_stream_width)
print(depth_img.shape, pressure_img.shape)

#%% Store data in a compressed numpy array file (npz)
depth_arr = np.zeros((1, depth_stream_height, depth_stream_width), dtype= 'uint16')
pressure_arr = np.zeros((1, pressure_stream_height, pressure_stream_width), dtype= 'float64')
pm_all = []

# ROWS 84, COLS 192, current shape 192,84
# ROWS and COLS spacing 1.016 cm
# SENSEL_AREA 1.03226 cm2
ROW_SPACING = 1.016 #cm
SENSEL_AREA = 1.03226/10000 # meter sq.
g = 9.80 #meter/sec
calc_weight_list = []
df = pd.read_csv(dir + '/physiqueData.csv')
weights = np.array(df["weight (kg)"])

avg_weight_diff_list = []

total_rows = len(names_df.index)
print("total subjects: ", len(names_df.index))
if total_rows>102:
    sys.exit(0)
with alive_bar(len(names_df.index)) as bar:
    for i, row in names_df.iterrows():
        depth_picture_path = row['Depth_path']
        pressure_picture_path = row['Pressure_path']
        pressure_cali_path = row['Pressure_cali_path']
        id = row['ids']

        for pic_path in depth_picture_path:
            depth_arr = np.concatenate( (depth_arr , np.load(pic_path)[np.newaxis,:]), axis=0)


        for pic_path in pressure_picture_path:
            pressure_arr = np.concatenate( (pressure_arr , np.load(pic_path)[np.newaxis,:]), axis=0)

        pm_data = np.load(pressure_cali_path)[2]
        pm_all.append(pm_data)

        avg_weight_list = []
        for j in range(1, 46):
            press = np.load(pic_path)
            press_calib= press*pm_data[j-1]
            taxel_weight_list = []
            for p in np.nditer(press_calib):
                if p>0:
                    mass = p*1000*SENSEL_AREA/g
                    taxel_weight_list.append(mass)
            weight_per_frame = np.sum(taxel_weight_list)
            avg_weight_list.append(weight_per_frame)
        avg_weight = abs(np.mean(avg_weight_list))
        calc_weight_list.append(avg_weight)
        # print("average weight of a person %d is %f compared to measured weight %f" %(i, avg_weight, weigths[i-1]))
        avg_weight_diff = abs(weights[i]-avg_weight)
        # print("difference between measured and calc weight: ", (avg_weight_diff))
        avg_weight_diff_list.append(avg_weight_diff)
        bar()
        # print(f"finished saving {i}th subject...")
    
pm_all = np.asarray(pm_all)
depth_arr = np.rot90(depth_arr[1:], k=-1, axes=(1,2))
pressure_arr = pressure_arr[1:]

fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(depth_arr[0])
ax[1].imshow(pressure_arr[0])
ax[0].set_title('depth image')
ax[1].set_title('pressure image')
fig.show()
plt.show()

print("depth array shape: ", depth_arr.shape)
print("pressure array shape: ", pressure_arr.shape)
print("pressure calibration shape", pm_all.shape)

######## save weight frame as csv ##########
avg_weight_diff = np.mean(avg_weight_diff_list)
print("average weight difference between measured and calcualted weight is %f"%avg_weight_diff)
wdf = pd.DataFrame({'gender':df["gender"]})
wdf['calculated weights (kg)'] = calc_weight_list
wdf['measured weights (kg)'] = weights
wdf.to_csv(npz_data_dir + '/weight_measurements.csv', index=False) 

########################## save filtered array ###############################
pm_adjust_mm = [12, -35]        # milimeter adjustment
pm_adjust_cm = [0, 0]
for i in range(2):
    if pm_adjust_mm[i] < 0:
        pm_adjust_cm[i] = int(float(pm_adjust_mm[i])/10. - 0.5)
    elif pm_adjust_mm[i] >= 0:
        pm_adjust_cm[i] = int(float(pm_adjust_mm[i])/10. + 0.5)
pm_adjust_cm[0] = int(-pm_adjust_cm[0])
pm_adjust_cm[1] = int(-pm_adjust_cm[1])


pressure_filtered, press_scale_fact = get_filtered_pm(pressure_arr, pm_adjust_cm, np.asarray(weights), SAVE_MMHG, USE_FILTER)
press_max = np.max(pressure_filtered)
pressure_filtered_arr = np.expand_dims(pressure_filtered/press_max, axis=3)
print("shape, min max after normalized pressure array: ", pressure_filtered_arr.shape, np.min(pressure_filtered_arr), np.max(pressure_filtered_arr))
print("max value in Pressure data: ", press_max)

fig, ax = plt.subplots(nrows=1,ncols=2)
ax[0].imshow(pressure_filtered[30])
ax[1].imshow(pressure_arr[30])
ax[0].set_title('after filter')
fig.show()
plt.show()

depth_max = np.max(depth_arr)
depth_arr = depth_arr/depth_max
print("shape, min max after normalized depth array: ", pressure_filtered_arr.shape, np.min(pressure_filtered_arr), np.max(pressure_filtered_arr))
print("max value in depth data: ", depth_max)

############# split dataset 60:20:20 #############
train_index = round(total_rows*0.6)
val_index = round(total_rows*0.6) + round(total_rows*0.2)
test_index = total_rows
print("train, val, test index: ", train_index, val_index, test_index)

x_train = depth_arr[:train_index*45]
x_val = depth_arr[train_index*45:val_index*45]
x_test = depth_arr[val_index*45:test_index*45]

y_train = pressure_filtered_arr[:train_index*45]
y_val = pressure_filtered_arr[train_index*45:val_index*45]
y_test = pressure_filtered_arr[val_index*45:test_index*45]

print("shape, min max after normalized x_train: ", x_train.shape, np.min(x_train), np.max(x_train))
print("shape, min max after normalized x_val: ", x_val.shape, np.min(x_val), np.max(x_val))
print("shape, min max after normalized x_test: ", x_test.shape, np.min(x_test), np.max(x_test))
print("shape, min max after normalized y_train: ", y_train.shape, np.min(y_train), np.max(y_train))
print("shape, min max after normalized y_val: ", y_val.shape, np.min(y_val), np.max(y_val))
print("shape, min max after normalized y_test: ", y_test.shape, np.min(y_test), np.max(y_test))

########## save compressed numpy array ##################
x_ttv = os.path.join(npz_data_dir, "x_ttv.npz").replace("\\","/")
y_ttv = os.path.join(npz_data_dir, "y_ttv.npz").replace("\\","/")
np.savez_compressed(x_ttv, x_train=x_train, x_val=x_val, x_test=x_test)
np.savez_compressed(y_ttv, y_train=y_train, y_val=y_val, y_test=y_test)

######### save caibration arrays #####################
train_calib = pm_all[:61].reshape(-1,1)
val_calib = pm_all[61:81].reshape(-1,1)
test_calib =  pm_all[81:].reshape(-1,1)
print("train calib scale shape, val calib scale shape, test calib scale shape", train_calib.shape, val_calib.shape, test_calib.shape)
np.save("train_press_calib_scale.npy",train_calib)
np.save("val_press_calib_scale.npy", val_calib)
np.save("test_press_calib_scale.npy", test_calib)

#%% Display Header
names_df.head()
names_df.to_csv('Dataframe.csv', index= False)