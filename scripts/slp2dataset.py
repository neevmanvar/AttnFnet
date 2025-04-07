import os
import glob
import csv
import sys
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from scipy.ndimage import gaussian_filter, zoom
import numpy as np
import pandas as pd

# Function: Apply filtering, calibration, and normalization to pressure mat data
# Source: https://github.com/Healthcare-Robotics/BodyPressure
# Function: get filtered pressure arrays
# License: MIT
# Taken from BodyPressure repository
# Copyright (c) Meta Platforms, Inc. and affiliates.
def get_filtered_pm(Arr, pm_adjust_cm, weights, save_KPa=False, use_filter=False, NORM_INDIVIDUAL=False):
    """
    Preprocess pressure mat data by cropping, smoothing, scaling, and optionally converting units.

    Parameters:
    - Arr (numpy.ndarray): 3D array of pressure mat images.
    - pm_adjust_cm (list): Adjustments to cropping in centimeters.
    - weights (numpy.ndarray): Array of subject weights.
    - save_KPa (bool): Flag to save data in KPa units.
    - use_filter (bool): Apply Gaussian smoothing filter if True.
    - NORM_INDIVIDUAL (bool): Normalize each image individually if True.

    Returns:
    - numpy.ndarray: Processed and rotated pressure mat data.
    - numpy.ndarray: Scaling factors used for normalization.
    """
    if Arr.ndim != 3:
        raise ValueError("array dimensions must be 3")

    scaling_factors = []
    filtered_arr = []

    for i, PM_arr in enumerate(Arr):
        # Adjust cropping based on provided parameters
        PM_arr = PM_arr[max(0, 1 - pm_adjust_cm[1]):191 - max(0, pm_adjust_cm[1]), max(0, 3 - pm_adjust_cm[0]):80 - max(0, pm_adjust_cm[0])]

        if use_filter:
            PM_arr = gaussian_filter(PM_arr, sigma=0.5, mode='reflect')

        PM_arr = zoom(PM_arr, (0.3355, 0.355), order=1)

        # print("before press conver", np.min(PM_arr), np.max(PM_arr))

        if save_KPa:
            MMHG_FACT = (1 / 133.322)
            KPA_FACT = (1 / 1000)
            PM_arr *= ((weights[i] * 9.81) / (np.sum(PM_arr) * 0.0264 * 0.0286)) * KPA_FACT

        if NORM_INDIVIDUAL:
            scale_factor = np.max(PM_arr)
            scaling_factors.append(scale_factor)
            PM_arr /= scale_factor

        filtered_arr.append(PM_arr)
        # print("after press conver", np.min(PM_arr), np.max(PM_arr))

    return np.rot90(np.asarray(filtered_arr), k=3, axes=(1, 2)), np.asarray(scaling_factors)

# Prompt user to optionally delete existing .npz files
x = input("Do you want to delete .npz files? (y/n): ")
npz_data_dir = "depth2bp_cleaned_no_KPa"
os.makedirs(npz_data_dir, exist_ok=True)
if x.lower() == 'y':
    [os.remove(file) for file in glob.glob(npz_data_dir+'/*.npz')]

USE_FILTER = True
SAVE_KPA = False

# Load dataset directory
DIR_NAME = 'danaLab'
ids = sorted(entry for entry in os.listdir(DIR_NAME) if os.path.isdir(os.path.join(DIR_NAME, entry)))
print("length of ids: ", len(ids)) # must be 102
print("ids: ", ids)    # must be 1 to 102

# Create DataFrame containing paths to depth, pressure, and calibration data
names_df = pd.DataFrame({
    'ids': ids,
    'Depth_path': [[f"{DIR_NAME}/{str(i).zfill(5)}/depthRaw/uncover/{str(j).zfill(6)}.npy" for j in range(1, 46)] for i in ids],
    'Pressure_path': [[f"{DIR_NAME}/{str(i).zfill(5)}/PMarray/uncover/{str(j).zfill(6)}.npy" for j in range(1, 46)] for i in ids],
    'Pressure_cali_path': [f"{DIR_NAME}/{str(i).zfill(5)}/PMcali.npy" for i in ids]
})

# Display full paths preview
pd.set_option('display.max_colwidth', None)
print(names_df.head())

# Initialize arrays based on the first image dimensions
sample_depth_img = np.load(names_df['Depth_path'][0][0])
depth_arr = np.zeros((1,) + sample_depth_img.shape, dtype='uint16')

sample_pressure_img = np.load(names_df['Pressure_path'][0][0])
pressure_arr = np.zeros((1,) + sample_pressure_img.shape, dtype='float64')

# Log image dimensions
print("depth height X width", sample_depth_img.shape)
print("pressure height X width", sample_pressure_img.shape)

# Load subject weights
df = pd.read_csv(f'{DIR_NAME}/physiqueData.csv')
weights = np.array(df["weight (kg)"])
pm_all = []
calc_weight_list = []
avg_weight_diff_list = []

# ROWS 84, COLS 192, current shape 192,84
# ROWS and COLS spacing 1.016 cm
# SENSEL_AREA 1.03226 cm2
ROW_SPACING = 1.016 #cm
SENSEL_AREA = 1.03226/10000 # meter sq.
g = 9.80 #meter/sec

total_rows = len(names_df)
print("total subjects: ", total_rows)
if total_rows > 102:
    sys.exit(0)

# Process all data
with alive_bar(total_rows) as bar:
    for i, row in names_df.iterrows():
        for pic_path in row['Depth_path']:
            depth_arr = np.concatenate((depth_arr, np.load(pic_path)[np.newaxis, :]))

        for pic_path in row['Pressure_path']:
            pressure_arr = np.concatenate((pressure_arr, np.load(pic_path)[np.newaxis, :]))

        pm_data = np.load(row['Pressure_cali_path'])[2]
        pm_all.append(pm_data)

        avg_weight_list = []
        for j, pic_path in enumerate(row['Pressure_path']):
            press_calib = np.load(pic_path) * pm_data[j]
            mass_per_frame = np.sum(press_calib[press_calib > 0] * 1000 * SENSEL_AREA / g)
            avg_weight_list.append(mass_per_frame)

        avg_weight = np.mean(avg_weight_list)
        calc_weight_list.append(avg_weight)
        avg_weight_diff = abs(weights[i] - avg_weight)
        avg_weight_diff_list.append(avg_weight_diff)
        bar()

# Save results and visualize
pm_all = np.asarray(pm_all)
depth_arr = np.rot90(depth_arr[1:], k=-1, axes=(1, 2))
pressure_arr = pressure_arr[1:]

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(depth_arr[0])
ax[1].imshow(pressure_arr[0])
ax[0].set_title('depth image')
ax[1].set_title('pressure image')
fig.show()
plt.show()

print("depth array shape: ", depth_arr.shape)
print("pressure array shape: ", pressure_arr.shape)
print("pressure calibration shape", pm_all.shape)

# Apply calibration adjustments
pm_adjust_mm = [12, -35]
pm_adjust_cm = [int(-round(mm / 10.)) for mm in pm_adjust_mm]
weights_arr = np.repeat(weights, 45)

pressure_filtered, press_scale_fact = get_filtered_pm(pressure_arr, pm_adjust_cm, weights_arr, SAVE_KPA, USE_FILTER)
press_max = np.max(pressure_filtered)
pressure_filtered_arr = np.expand_dims(pressure_filtered / press_max, axis=3)
print("shape, min max after normalized pressure array: ", pressure_filtered_arr.shape, np.min(pressure_filtered_arr), np.max(pressure_filtered_arr))
print("max value in Pressure data: ", press_max)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(pressure_filtered[30])
ax[1].imshow(pressure_arr[30])
ax[0].set_title('after filter')
fig.show()
plt.show()

depth_max = np.max(depth_arr)
depth_arr = depth_arr / depth_max
print("shape, min max after normalized depth array: ", pressure_filtered_arr.shape, np.min(pressure_filtered_arr), np.max(pressure_filtered_arr))
print("max value in depth data: ", depth_max)

# Split data into training, validation, and test sets
train_index = round(total_rows * 0.6)
val_index = round(total_rows * 0.6) + round(total_rows * 0.2)
test_index = total_rows
print("train, val, test index: ", train_index, val_index, test_index)

x_train = depth_arr[:train_index*45]
x_val = depth_arr[train_index*45:val_index*45]
x_test = depth_arr[val_index*45:test_index*45]

y_train = pressure_filtered_arr[:train_index*45]
y_val = pressure_filtered_arr[train_index*45:val_index*45]
y_test = pressure_filtered_arr[val_index*45:test_index*45]

f = open(f"{npz_data_dir}Scaling_factors_global.txt", "w")
f.write('\n')
f.write("depth array factor = " + str(depth_max))
f.write('\n')
f.write("pressure array factor = " + str(press_max))
f.close()

print("shape, min max after normalized x_train: ", x_train.shape, np.min(x_train), np.max(x_train))
print("shape, min max after normalized x_val: ", x_val.shape, np.min(x_val), np.max(x_val))
print("shape, min max after normalized x_test: ", x_test.shape, np.min(x_test), np.max(x_test))
print("shape, min max after normalized y_train: ", y_train.shape, np.min(y_train), np.max(y_train))
print("shape, min max after normalized y_val: ", y_val.shape, np.min(y_val), np.max(y_val))
print("shape, min max after normalized y_test: ", y_test.shape, np.min(y_test), np.max(y_test))

# Save datasets
np.savez_compressed(f"{npz_data_dir}/x_ttv.npz", x_train=x_train, x_val=x_val, x_test=x_test)
np.savez_compressed(f"{npz_data_dir}/y_ttv.npz", y_train=y_train, y_val=y_val, y_test=y_test)

# Save calibration arrays
train_calib = pm_all[:train_index].reshape(-1, 1)
val_calib = pm_all[train_index:val_index].reshape(-1, 1)
test_calib = pm_all[val_index:].reshape(-1, 1)
print("train calib scale shape, val calib scale shape, test calib scale shape", train_calib.shape, val_calib.shape, test_calib.shape)
np.save(f"{npz_data_dir}/train_press_calib_scale.npy", train_calib)
np.save(f"{npz_data_dir}/val_press_calib_scale.npy", val_calib)
np.save(f"{npz_data_dir}/test_press_calib_scale.npy", test_calib)

# Display header and save dataframe
names_df.head()
names_df.to_csv(f'{npz_data_dir}Dataframe.csv', index=False)

# Save weight calculations to CSV
avg_weight_diff = np.mean(avg_weight_diff_list)
print("average weight difference between measured and calcualted weight is %f" % avg_weight_diff)
wdf = pd.DataFrame({'gender': df["gender"]})
wdf['calculated weights (kg)'] = calc_weight_list
wdf['measured weights (kg)'] = weights
wdf.to_csv(f"{npz_data_dir}/weight_measurements.csv", index=False)
