import numpy as np
import scipy.stats
import glob
import csv
import h5py
import pandas as pd
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis


# predefine inputs
root_dir = '/media/lucab/data_hdd/screen_corners'
behav_dir = '/media/lucab/data_hdd/CommGame_behav'
pairNo_range = (31, 110)  # both values are inclusive!
session_names = ['BG1', 'BG2', 'BG3', 'BG4']
#session_names = ['BG1', 'BG2', 'BG3', 'BG4', 'freeConv']

# blacklist = ('/prepro_pl_data_pair36_Mordor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair36_Mordor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair36_Mordor_BG3_residual.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG3_residual.hdf5',
#              '/prepro_pl_data_pair37_Mordor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair37_Gondor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair39_Gondor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair39_Gondor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair41_Gondor_BG5_residual.hdf5',
#              '/prepro_pl_data_pair41_Gondor_BG6_residual.hdf5',
#              '/prepro_pl_data_pair45_Mordor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair49_Gondor_BG4_residual.hdf5',
#              '/prepro_pl_data_pair54_Mordor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair54_Mordor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair59_Mordor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair59_Gondor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair59_Gondor_BG3_residual.hdf5',
#              '/prepro_pl_data_pair60_Mordor_BG4_residual.hdf5',
#              '/prepro_pl_data_pair60_Gondor_BG3_residual.hdf5',
#              '/prepro_pl_data_pair62_Gondor_BG5_residual.hdf5',
#              '/prepro_pl_data_pair62_Gondor_BG6_residual.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG3_residual.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG4_residual.hdf5',
#              '/prepro_pl_data_pair77_Gondor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair91_Mordor_BG1_residual.hdf5',
#              '/prepro_pl_data_pair91_Mordor_BG2_residual.hdf5',
#              '/prepro_pl_data_pair36_Mordor_freeConv_residual.hdf5',
#              '/prepro_pl_data_pair36_Gondor_freeConv_residual.hdf5',
#              '/prepro_pl_data_pair37_Mordor_freeConv_residual.hdf5',
#              '/prepro_pl_data_pair59_Mordor_freeConv_residual.hdf5',
#              '/prepro_pl_data_pair36_Mordor_BG1.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG1.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG2.hdf5',
#              '/prepro_pl_data_pair36_Gondor_BG3.hdf5',
#              '/prepro_pl_data_pair37_Mordor_BG1.hdf5',
#              '/prepro_pl_data_pair37_Gondor_BG2.hdf5',
#              '/prepro_pl_data_pair39_Gondor_BG1.hdf5',
#              '/prepro_pl_data_pair39_Gondor_BG2.hdf5',
#              '/prepro_pl_data_pair41_Gondor_BG5.hdf5',
#              '/prepro_pl_data_pair41_Gondor_BG6.hdf5',
#              '/prepro_pl_data_pair45_Mordor_BG2.hdf5',
#              '/prepro_pl_data_pair49_Gondor_BG4.hdf5',
#              '/prepro_pl_data_pair54_Mordor_BG1.hdf5',
#              '/prepro_pl_data_pair54_Mordor_BG2.hdf5',
#              '/prepro_pl_data_pair59_Mordor_BG2.hdf5',
#              '/prepro_pl_data_pair59_Gondor_BG1.hdf5',
#              '/prepro_pl_data_pair59_Gondor_BG3.hdf5',
#              '/prepro_pl_data_pair60_Mordor_BG4.hdf5',
#              '/prepro_pl_data_pair60_Gondor_BG3.hdf5',
#              '/prepro_pl_data_pair62_Gondor_BG5.hdf5',
#              '/prepro_pl_data_pair62_Gondor_BG6.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG2.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG3.hdf5',
#              '/prepro_pl_data_pair64_Gondor_BG4.hdf5',
#              '/prepro_pl_data_pair77_Gondor_BG2.hdf5',
#              '/prepro_pl_data_pair91_Mordor_BG1.hdf5',
#              '/prepro_pl_data_pair91_Mordor_BG2.hdf5',
#              '/prepro_pl_data_pair36_Mordor_freeConv.hdf5',
#              '/prepro_pl_data_pair36_Gondor_freeConv.hdf5',
#              '/prepro_pl_data_pair37_Mordor_freeConv.hdf5',
#              '/prepro_pl_data_pair59_Mordor_freeConv.hdf5'
#              )


blacklist = ('/prepro_pl_data_pair36_Mordor_BG1_residual.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG1_residual.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG2_residual.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG2_residual.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG3_residual.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG3_residual.hdf5',
             '/prepro_pl_data_pair36_Mordor_freeConv_residual.hdf5',
             '/prepro_pl_data_pair36_Gondor_freeConv_residual.hdf5',
             '/prepro_pl_data_pair37_Mordor_freeConv_residual.hdf5',
             '/prepro_pl_data_pair59_Mordor_freeConv_residual.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG1.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG1.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG2.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG2.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG3.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG3.hdf5',
             '/prepro_pl_data_pair36_Mordor_freeConv.hdf5',
             '/prepro_pl_data_pair36_Gondor_freeConv.hdf5',
             '/prepro_pl_data_pair37_Mordor_freeConv.hdf5',
             '/prepro_pl_data_pair59_Mordor_freeConv.hdf5'
             )

def extract_condition(all_pairs, cond):

    csv_file = "/home/lucab/Documents/commgame_conditions.csv"
    cond_df = pd.read_csv(csv_file)
    tmp = cond_df.loc[cond_df['Condition'] == cond]
    cond_extracted = tmp.loc[tmp['Pair'].between(all_pairs[0], all_pairs[1])]
    pairs = np.array(np.unique(cond_extracted['Pair']))
    print(pairs)

    return pairs

def data_collect(root_dir, pairNo_range, session_names):

    pairNos = range(pairNo_range[0], pairNo_range[1] + 1)
    labNames = ["Mordor", "Gondor"]

    mordor_files = []
    mordor_ids_pairNo = []
    mordor_ids_session = []

    for pairNo in pairNos:
            for session in session_names:

                mordor_file = glob.glob(f'{root_dir}/**/prepro_pl_data_pair{pairNo}_Mordor_{session}_residual.hdf5', recursive=True)
                if len(mordor_file) == 0:
                    mordor_file = ['Invalid']
                mordor_files.append(mordor_file[0])
                mordor_ids_pairNo.append(pairNo)
                mordor_ids_session.append(session)

    gondor_files = []
    gondor_ids_pairNo = []
    gondor_ids_session = []

    for pairNo in pairNos:
            for session in session_names:
                gondor_file = glob.glob(f'{root_dir}/**/prepro_pl_data_pair{pairNo}_Gondor_{session}_residual.hdf5',
                                        recursive=True)
                if len(gondor_file) == 0:
                    gondor_file = ['Invalid']
                gondor_files.append(gondor_file[0])
                gondor_ids_pairNo.append(pairNo)
                gondor_ids_session.append(session)
                gondor_mat_file = glob.glob(f'{behav_dir}/**/pair{pairNo}_Gondor_behav/pair{pairNo}_Gondor_{session}_**imes.mat',
                              recursive=True)




    # check for asymmetry in input data
    if not len(mordor_files) == len(gondor_files):
        print('\n!!!WARNING!!! '
              '\nFound an asymmetry in the number of recordings across labs!!!'
              '\n!!!WARNING!!!\n'
              '\nNumber of files for Mordor: {}'
              '\nNUmber of files for Gondor: {}' .format(len(mordor_files), len(gondor_files)))

    print(mordor_files, gondor_files)

    #paired_files = np.column_stack((mordor_files, gondor_files))
    paired_files = list(zip(mordor_files, gondor_files))
    paired_ids_pairNo = np.column_stack((mordor_ids_pairNo, gondor_ids_pairNo))
    paired_ids_session = np.column_stack((mordor_ids_session, gondor_ids_session))

    return paired_files, paired_ids_pairNo, paired_ids_session


def h5_reader(filepath):

    if filepath == 'Invalid':
        print("\n Not a valid path, all data will be NaN!")
        pupil_timestamps = np.nan
        pupil_diameter = np.nan
        pupil_norm_pos_x = np.nan
        pupil_norm_pos_y = np.nan
        pupil_isnan = np.nan

    else:
        print('\nReading {} ...' .format(filepath))

        f = h5py.File(filepath, 'r')
        f.visititems(print)

        for key in f.keys():
            pair = f[key]
            for lab in pair.keys():
                labName = pair[lab]
                for sess in labName.keys():
                    session = labName[sess]
                    for key in session.attrs.keys():
                        print('{} : {}' .format(key, session.attrs[key]))
                    pupil_timestamps = np.array(session['pupil_timestamps'])
                    pupil_diameter = np.array(session['pupil_diameter'])
                    pupil_norm_pos_x = np.array(session['pupil_norm_pos_x'])
                    pupil_norm_pos_y = np.array(session['pupil_norm_pos_y'])
                    pupil_isnan = np.array(session['pupil_isnan'])
                    #pupil_brightness = session['pupil_brightness']

    #f.close()

    return pupil_timestamps, pupil_diameter, pupil_norm_pos_x, pupil_norm_pos_y, pupil_isnan


def downsample_pupil(timestamps, diameter, isnan, sample_rate=30):

    n = int(120/sample_rate)
    s = slice(0, None, n)
    diameter_downsampled = diameter[s]
    timestamps_downsampled = timestamps[s]
    isnan_downsampled = isnan[s]

    return timestamps_downsampled, diameter_downsampled, isnan_downsampled


def compute_dtw(series_one, series_two, series_one_nan, series_two_nan, sample_rate=30, window_size=6, step_size=3):

    window_starts = np.arange(0, round(np.max(series_one.shape) - sample_rate * window_size), round(sample_rate * step_size))
    window_ends = np.arange(sample_rate * window_size, np.max(series_one.shape), round(sample_rate * step_size))
    dtw_windows = []

    for segment, start in enumerate(window_starts):
        array_segment_one = series_one[start: window_ends[segment]]
        array_segment_two = series_two[start: window_ends[segment]]

        # check for nan ratio in segment
        nans_segment_one = series_one_nan[start: window_ends[segment]]
        segment_one_nanratio = sum(nans_segment_one) / len(nans_segment_one) * 100
        nans_segment_two = series_two_nan[start: window_ends[segment]]
        segment_two_nanratio = sum(nans_segment_two) / len(nans_segment_two) * 100

        if segment_one_nanratio >= 40 or segment_two_nanratio >= 40:
            distance = np.nan
        else:
            # compute dtw
            #distance = dtw.distance_fast(array_segment_one, array_segment_two, window=90, max_step=30)
            distance = dtw.distance_fast(array_segment_one, array_segment_two)

        dtw_windows.append(distance)

    mean_dist = np.nanmean(dtw_windows)
    median_dist = np.nanmedian(dtw_windows)

    return dtw_windows, mean_dist, median_dist



# MAIN PART

dtw_mean_results = []
dtw_window_results = []
# get pair numbers according to condition type
# pairNos = extract_condition(all_pairs=pairNo_range, cond="ismeretlen")
# collect filenames
paired_files, paired_ids_pairNo, paired_ids_session = data_collect(root_dir, pairNo_range, session_names)

for paired_file in paired_files:

    # check for missing files
    if paired_file[0] == 'Invalid' or paired_file[1] == 'Invalid':
        print("Will skip this pair, missing / corrupt files!")
        dtw_value = np.nan
        dtw_mean_results.append(dtw_value)
        dtw_window_values = [np.nan] * 10  # want to mimic the real data, so create a list with multiple nans
        dtw_window_results.append(dtw_window_values)
        continue
    if paired_file[0].endswith(blacklist) or paired_file[1].endswith(blacklist):
        print('Blacklist file!')
        dtw_value = np.nan
        dtw_mean_results.append(dtw_value)
        dtw_window_values = [np.nan] * 10
        dtw_window_results.append(dtw_window_values)
        continue


    # read hdf5 file for mordor and for gondor
    else:
        mordor_timestamps, mordor_diameter, mordor_norm_pos_x, mordor_norm_pos_y, mordor_isnan = h5_reader(paired_file[0])
        gondor_timestamps, gondor_diameter, gondor_norm_pos_x, gondor_norm_pos_y, gondor_isnan = h5_reader(paired_file[1])

        # # downsample to 30Hz
        # mordor_timestamps, mordor_diameter, mordor_isnan = downsample_pupil(mordor_timestamps, mordor_diameter,
        #                                                                     mordor_isnan, sample_rate=30)
        # gondor_timestamps, gondor_diameter, gondor_isnan = downsample_pupil(gondor_timestamps, gondor_diameter,
        #                                                                     gondor_isnan, sample_rate=30)

        # perform z-scoring
        mordor_diameter = scipy.stats.zscore(mordor_diameter, ddof=0, nan_policy='omit')
        gondor_diameter = scipy.stats.zscore(gondor_diameter, ddof=0, nan_policy='omit')

        # trim to same size - not sure if we need it at this point
        if np.size(mordor_diameter) < np.size(gondor_diameter):
            gondor_diameter = gondor_diameter[0:np.size(mordor_diameter)]
            gondor_timestamps = gondor_timestamps[0:np.size(mordor_timestamps)]
            gondor_isnan = gondor_isnan[0:np.size(mordor_isnan)]
        if np.size(gondor_diameter) < np.size(mordor_diameter):
            mordor_diameter = mordor_diameter[0:np.size(gondor_diameter)]
            mordor_timestamps = mordor_timestamps[0:np.size(gondor_timestamps)]
            mordor_isnan = mordor_isnan[0:np.size(gondor_isnan)]

        # calculate DTW
        dtw_windows, mean_dist, median_dist = compute_dtw(mordor_diameter, gondor_diameter, mordor_isnan, gondor_isnan)

        dtw_mean_results.append(mean_dist)
        dtw_window_results.append(dtw_windows)

print(dtw_mean_results)
print(len(dtw_mean_results))
print(len(paired_files))

pair_numbers = np.array(np.unique(paired_ids_pairNo))
sessions = np.array(np.unique(session_names))
tmp = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], dtw_mean_results))

# create dataframe
dtw_results_df_long = pd.DataFrame(tmp, columns=["Pair Number", "Session", "DTW mean"])
# for some reason the data is in str (?) - need to convert it to float
dtw_results_df_wide = dtw_results_df_long.pivot(index="Pair Number", columns="Session", values="DTW mean")


dtw_results_df_wide = dtw_results_df_wide.to_numpy()
# for some reason the data is in str (?) - need to convert it to float
dtw_results_df_wide = dtw_results_df_wide.astype(float)
dtw_results_df_long = dtw_results_df_long.to_numpy()

print('\nMean distance: {}' .format(np.nanmean(dtw_mean_results)))
#print('Mean distance for FreeConv: {}' .format(np.nanmean(dtw_results_df_wide[:, 4])))
#print('Mean distance for BG: {}' .format(np.nanmean(dtw_results_df_wide[:, 0:3])))

np.save('pupil_dtw_wide_res_zscored_63_BG', dtw_results_df_wide)
np.save('pupil_dtw_long_res_zscored_63_BG', dtw_results_df_long)


length = max(map(len, dtw_window_results))
dtw_array = np.array([i+[np.nan]*(length-len(i)) for i in dtw_window_results])
dtw_array = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], dtw_array))
print(dtw_array)
print(type(dtw_array[0]))
print(np.shape(dtw_array))

np.save('dtw_window_result_BG', dtw_array)

# with open("dtw_window_results_BG.csv", "w") as f:
#     csv_file = csv.writer(f)
#     csv_file.writerows(dtw_list)

