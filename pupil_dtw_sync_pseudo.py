import numpy as np
import glob
import h5py
import scipy
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
from scipy import stats


# predefine inputs
root_dir = '/media/lucab/data_hdd/CommGame_pupil_resampled'
# session_names = ['BG1', 'BG2', 'BG3', 'BG4']
cond = 'unimodális'

# skip these files
blacklist = ('/prepro_pl_data_pair36_Mordor_BG1_residual_resampled.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG1_residual_resampled.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG2_residual_resampled.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG2_residual_resampled.hdf5',
             '/prepro_pl_data_pair36_Mordor_BG3_residual_resampled.hdf5',
             '/prepro_pl_data_pair36_Gondor_BG3_residual_resampled.hdf5',
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



def data_collect(root_dir, cond):

    # loading npy file containing the pseudo pairs
    if cond == 'alap':
        pseudo_pairs = np.load("/media/lucab/data_hdd/pseudo_pairs_list_new_fam.npy")
    if cond == 'ismeretlen':
        pseudo_pairs = np.load("/media/lucab/data_hdd/pseudo_pairs_list_new_unfam.npy")
    if cond == 'unimodális':
        pseudo_pairs = np.load("/media/lucab/data_hdd/pseudo_pairs_list_new_uni.npy")

    pairNos_mordor = pseudo_pairs[:, 1]
    pairNos_gondor = pseudo_pairs[:, 2]

    mordor_files = []
    mordor_ids_pairNo = []
    mordor_ids_session = []

    for idx, pairNo in enumerate(pairNos_mordor):
        if pseudo_pairs[idx, 0] == 4:
            session = 'BG4'
        if pseudo_pairs[idx, 0] == 3:
            session = 'BG3'
        if pseudo_pairs[idx, 0] == 2:
            session = 'BG2'
        if pseudo_pairs[idx, 0] == 1:
            session = 'BG1'
        pairNo = int(pairNo)
        mordor_file = glob.glob(f'{root_dir}/**/prepro_pl_data_pair{pairNo}_Mordor_{session}_residual_resampled.hdf5', recursive=True)
        if len(mordor_file) == 0:
            mordor_file = ['Invalid']
        mordor_files.append(mordor_file[0])
        mordor_ids_pairNo.append(pairNo)
        mordor_ids_session.append(session)


    gondor_files = []
    gondor_ids_pairNo = []
    gondor_ids_session = []

    for idx, pairNo in enumerate(pairNos_gondor):
        if pseudo_pairs[idx, 0] == 4:
            session = 'BG4'
        if pseudo_pairs[idx, 0] == 3:
            session = 'BG3'
        if pseudo_pairs[idx, 0] == 2:
            session = 'BG2'
        if pseudo_pairs[idx, 0] == 1:
            session = 'BG1'
        pairNo = int(pairNo)
        gondor_file = glob.glob(f'{root_dir}/**/prepro_pl_data_pair{pairNo}_Gondor_{session}_residual_resampled.hdf5', recursive=True)
        if len(gondor_file) == 0:
            gondor_file = ['Invalid']
        gondor_files.append(gondor_file[0])
        gondor_ids_pairNo.append(pairNo)
        gondor_ids_session.append(session)


    # check for asymmetry in input data
    if not len(mordor_files) == len(gondor_files):
        print('\n!!!WARNING!!! '
              '\nFound an asymmetry in the number of recordings across labs!!!'
              '\n!!!WARNING!!!\n'
              '\nNumber of files for Mordor: {}'
              '\nNUmber of files for Gondor: {}' .format(len(mordor_files), len(gondor_files)))


    paired_files = list(zip(mordor_files, gondor_files))
    print(paired_files)
    paired_ids_pairNo = np.column_stack((mordor_ids_pairNo, gondor_ids_pairNo))
    print(paired_ids_pairNo)
    paired_ids_session = np.column_stack((mordor_ids_session, gondor_ids_session))
    print(paired_ids_session)

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
                    pupil_timestamps = np.array(session['pupil_timestamps_resampled'])
                    pupil_diameter = np.array(session['pupil_diameter_resampled'])
                    pupil_norm_pos_x = np.array(session['pupil_norm_pos_x_resampled'])
                    pupil_norm_pos_y = np.array(session['pupil_norm_pos_y_resampled'])
                    pupil_isnan = np.array(session['pupil_isnan_resampled'])
                    #pupil_brightness = session['pupil_brightness']

    #f.close()

    return pupil_timestamps, pupil_diameter, pupil_norm_pos_x, pupil_norm_pos_y, pupil_isnan


# def downsample_pupil(timestamps, diameter, isnan, sample_rate=30):
#
#     n = int(120/sample_rate)
#     s = slice(0, None, n)
#     diameter_downsampled = diameter[s]
#     timestamps_downsampled = timestamps[s]
#     isnan_downsampled = isnan[s]
#
#     return timestamps_downsampled, diameter_downsampled, isnan_downsampled


def compute_dtw(series_one, series_two, series_one_nan, series_two_nan, sample_rate=30, window_size=6, step_size=3):

    window_starts = np.arange(0, round(np.max(series_one.shape) - sample_rate * window_size), round(sample_rate * step_size))
    window_ends = np.arange(sample_rate * window_size, np.max(series_one.shape), round(sample_rate * step_size))
    dtw_windows = []
    nan_ratios_one = []
    nan_ratios_two = []

    for segment, start in enumerate(window_starts):
        array_segment_one = series_one[start: window_ends[segment]]
        array_segment_two = series_two[start: window_ends[segment]]

        if len(array_segment_one) == 0 or len(array_segment_two) == 0:
            distance = np.nan
            segment_one_nanratio = np.nan
            segment_two_nanratio = np.nan
        else:
            # check for nan ratio in segment
            nans_segment_one = series_one_nan[start: window_ends[segment]]
            segment_one_nanratio = sum(nans_segment_one) / len(nans_segment_one) * 100
            nans_segment_two = series_two_nan[start: window_ends[segment]]
            segment_two_nanratio = sum(nans_segment_two) / len(nans_segment_two) * 100

            # compute dtw
            #distance = dtw.distance_fast(array_segment_one, array_segment_two, window=90, max_step=30)
            distance = dtw.distance_fast(array_segment_one, array_segment_two)

        dtw_windows.append(distance)
        nan_ratios_one.append(segment_one_nanratio)
        nan_ratios_two.append(segment_two_nanratio)

    mean_dist = np.nanmean(dtw_windows)
    median_dist = np.nanmedian(dtw_windows)

    return dtw_windows, mean_dist, median_dist, nan_ratios_one, nan_ratios_two


# MAIN PART

dtw_mean_results = []
dtw_window_results = []
nan_ratios_mordor_all = []
nan_ratios_gondor_all = []
paired_files, paired_ids_pairNo, paired_ids_session = data_collect(root_dir, cond)

for paired_file in paired_files:

    # check for missing files
    if paired_file[0] == 'Invalid' or paired_file[1] == 'Invalid':
        print("Will skip this pair, missing / corrupt files!")
        dtw_value = np.nan
        dtw_mean_results.append(dtw_value)
        dtw_window_values = [np.nan] * 10  # want to mimic the real data, so create a list with multiple nans
        dtw_window_results.append(dtw_window_values)
        nan_ratios_mordor_all.append(dtw_window_values)
        nan_ratios_gondor_all.append(dtw_window_values)
        continue
    if paired_file[0].endswith(blacklist) or paired_file[1].endswith(blacklist):
        print('Blacklist file!')
        dtw_value = np.nan
        dtw_mean_results.append(dtw_value)
        dtw_window_values = [np.nan] * 10
        dtw_window_results.append(dtw_window_values)
        nan_ratios_mordor_all.append(dtw_window_values)
        nan_ratios_gondor_all.append(dtw_window_values)
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

        # # trim to same size
        # if np.size(mordor_diameter) < np.size(gondor_diameter):
        #     gondor_diameter = gondor_diameter[0:np.size(mordor_diameter)]
        #     gondor_timestamps = gondor_timestamps[0:np.size(mordor_timestamps)]
        #     gondor_isnan = gondor_isnan[0:np.size(mordor_isnan)]
        # if np.size(gondor_diameter) < np.size(mordor_diameter):
        #     mordor_diameter = mordor_diameter[0:np.size(gondor_diameter)]
        #     mordor_timestamps = mordor_timestamps[0:np.size(gondor_timestamps)]
        #     mordor_isnan = mordor_isnan[0:np.size(gondor_isnan)]

        # Compute DTW
        dtw_windows, mean_dist, median_dist, nan_ratios_mordor, nan_ratios_gondor = compute_dtw(mordor_diameter,
                                                                                                gondor_diameter,
                                                                                                mordor_isnan,
                                                                                                gondor_isnan)

        dtw_mean_results.append(mean_dist)
        dtw_window_results.append(dtw_windows)
        nan_ratios_mordor_all.append(nan_ratios_mordor)
        nan_ratios_gondor_all.append(nan_ratios_gondor)

print(len(dtw_mean_results))
print(len(paired_ids_pairNo), len(paired_ids_session))

# pair_numbers = np.array(np.unique(paired_ids_pairNo))
# sessions = np.array(np.unique(session_names))
df_tmp = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], dtw_mean_results))
print(dtw_mean_results)

print('\nDTW mean for pseudo pairs: {}. Condition: {}' .format(np.nanmean(dtw_mean_results), cond))


# Now saving the results of each segment for each pair
# convert to a format where each row is one participant's one session and each segment is in a separate column
length = max(map(len, dtw_window_results))
dtw_array = np.array([i+[np.nan]*(length-len(i)) for i in dtw_window_results])

# also for the nan ratios
length_nan = max(map(len, nan_ratios_mordor_all))
nan_ratios_mordor_all_new = np.array([m+[np.nan]*(length_nan-len(m)) for m in nan_ratios_mordor_all])
length_nan = max(map(len, nan_ratios_gondor_all))
nan_ratios_gondor_all_new = np.array([g+[np.nan]*(length_nan-len(g)) for g in nan_ratios_gondor_all])
nan_ratios_3D = np.stack((nan_ratios_mordor_all_new, nan_ratios_gondor_all_new), axis=1)

# put it all in an array with the pair number and session id-s
dtw_array = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], dtw_array))
nans_array_mordor = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], nan_ratios_mordor_all_new))
nans_array_gondor = np.column_stack((paired_ids_pairNo[:, 0], paired_ids_session[:, 0], nan_ratios_gondor_all_new))

if cond == 'alap':
    np.save('/home/lucab/Documents/pseudo_pupil_dtw_zscored_resampled_BG_fam_67', dtw_mean_results)
    np.save('dtw_window_result_resampled_BG_fam_pseudo_67', dtw_array)
    np.save('mordor_nan_ratios_windows_resampled_BG_fam_pseudo_67', nans_array_mordor)
    np.save('gondor_nan_ratios_windows_resampled_BG_fam_pseudo_67', nans_array_gondor)
if cond == 'ismeretlen':
    np.save('/home/lucab/Documents/pseudo_pupil_dtw_zscored_resampled_BG_unfam_67', dtw_mean_results)
    np.save('dtw_window_result_resampled_BG_unfam_pseudo_67', dtw_array)
    np.save('mordor_nan_ratios_windows_resampled_BG_unfam_pseudo_67', nans_array_mordor)
    np.save('gondor_nan_ratios_windows_resampled_BG_unfam_pseudo_67', nans_array_gondor)
if cond == 'unimodális':
    np.save('/home/lucab/Documents/pseudo_pupil_dtw_zscored_resampled_BG_uni_67', dtw_mean_results)
    np.save('dtw_window_result_resampled_BG_uni_pseudo_67', dtw_array)
    np.save('mordor_nan_ratios_windows_resampled_BG_uni_pseudo_67', nans_array_mordor)
    np.save('gondor_nan_ratios_windows_resampled_BG_uni_pseudo_67', nans_array_gondor)


