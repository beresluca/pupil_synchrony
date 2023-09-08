import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from math import sqrt

cond = 'unimodális'  # if this value is set to 'None', all conditions are considered
cutoff_primary = 50  # segment is discarded if ratio of NaNs is above this threshold
cutoff_secondary = 50  # recording is kept if VALID segments' ratio is above this threshold

# Load data
if cond == 'alap':
    real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_fam_67_nonreg.npy')
    pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_fam_pseudo_67_nonreg.npy')
    nan_gondor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_fam_pseudo_67_nonreg.npy')
    nan_gondor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_fam_67_nonreg.npy')
    nan_mordor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_fam_pseudo_67_nonreg.npy')
    nan_mordor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_fam_67_nonreg.npy')
if cond == 'ismeretlen':
    real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_unfam_67_nonreg.npy')
    pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_unfam_pseudo_67_nonreg.npy')
    nan_gondor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_unfam_pseudo_67_nonreg.npy')
    nan_gondor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_unfam_67_nonreg.npy')
    nan_mordor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_unfam_pseudo_67_nonreg.npy')
    nan_mordor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_unfam_67_nonreg.npy')
if cond == 'unimodális':
    real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_uni_67_nonreg.npy')
    pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_uni_pseudo_67_nonreg.npy')
    nan_gondor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_uni_pseudo_67_nonreg.npy')
    nan_gondor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_uni_67_nonreg.npy')
    nan_mordor_pseudo_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_uni_pseudo_67_nonreg.npy')
    nan_mordor_real_file = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_uni_67_nonreg.npy')
if cond == None:
    real_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG4_fam_67.npy')
    pseudo_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_fam_pseudo.npy')
    nan_gondor_pseudo_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_fam_pseudo.npy')
    nan_gondor_real_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG4_fam_67.npy')
    nan_mordor_pseudo_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_fam_pseudo.npy')
    nan_mordor_real_file1 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG4_fam_67.npy')
    real_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG4_unfam_67.npy')
    pseudo_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_unfam_pseudo.npy')
    nan_gondor_pseudo_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_unfam_pseudo.npy')
    nan_gondor_real_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG4_unfam_67.npy')
    nan_mordor_pseudo_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_unfam_pseudo.npy')
    nan_mordor_real_file2 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG4_unfam_67.npy')
    real_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG4_uni_67.npy')
    pseudo_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/dtw_window_result_resampled_BG_uni_pseudo.npy')
    nan_gondor_pseudo_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG_uni_pseudo.npy')
    nan_gondor_real_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/gondor_nan_ratios_windows_resampled_BG4_uni_67.npy')
    nan_mordor_pseudo_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG_uni_pseudo.npy')
    nan_mordor_real_file3 = np.load('/home/lucab/PycharmProjects/pupil_postprocessing/mordor_nan_ratios_windows_resampled_BG4_uni_67.npy')
    # create one array from all
    real_file = np.concatenate((real_file1, real_file2, real_file3))
    pseudo_file = np.concatenate((pseudo_file1, pseudo_file2, pseudo_file3))
    nan_gondor_pseudo_file = np.concatenate((nan_gondor_pseudo_file1, nan_gondor_pseudo_file2, nan_gondor_pseudo_file3))
    nan_gondor_real_file = np.concatenate((nan_gondor_real_file1, nan_gondor_real_file2, nan_gondor_real_file3))
    nan_mordor_pseudo_file = np.concatenate((nan_mordor_pseudo_file1, nan_mordor_pseudo_file2, nan_mordor_pseudo_file3))
    nan_mordor_real_file = np.concatenate((nan_mordor_real_file1, nan_mordor_real_file2, nan_mordor_real_file3))


# housekeeping
real_pairs = real_file[:, 2:].astype(float)
pseudo_pairs = pseudo_file[:, 2:].astype(float)
nan_gondor_pseudo = nan_gondor_pseudo_file[:, 2:].astype(float)
nan_gondor_real = nan_gondor_real_file[:, 2:].astype(float)
nan_mordor_pseudo = nan_mordor_pseudo_file[:, 2:].astype(float)
nan_mordor_real = nan_mordor_real_file[:, 2:].astype(float)

# new = np.zeros((len(real_pairs[0]) / 4, 1))


# FIRST STEP

# remove segments based on nan ratios

real_pairs_segments_removed = np.copy(real_pairs)
pseudo_pairs_segments_removed = np.copy(pseudo_pairs)

for ((real_value_idx, real_value),
     (nan_value_m_idx, nan_value_m),
     (nan_value_g_idx, nan_value_g)) in zip(np.ndenumerate(real_pairs_segments_removed),
                                            np.ndenumerate(nan_mordor_real),
                                            np.ndenumerate(nan_gondor_real)):
    if real_value != np.nan:
        if nan_value_m >= cutoff_primary or nan_value_g >= cutoff_primary:
            real_pairs_segments_removed[real_value_idx] = np.nan


for ((pseudo_value_idx, pseudo_value),
     (nan_value_m_idx, nan_value_m),
     (nan_value_g_idx, nan_value_g)) in zip(np.ndenumerate(pseudo_pairs_segments_removed),
                                            np.ndenumerate(nan_mordor_pseudo),
                                            np.ndenumerate(nan_gondor_pseudo)):
    if pseudo_value != np.nan:
        if nan_value_m >= cutoff_primary or nan_value_g >= cutoff_primary:
            pseudo_pairs_segments_removed[pseudo_value_idx] = np.nan


print(np.array_equal(real_pairs, real_pairs_segments_removed))


# SECOND STEP

# removing recordings with very few remaining segments

# for real pairs

# getting the last index where data is not nan (in a row),
# to know how many segments we have in one recording
sum_of_segments_list = []
for recording in real_pairs:
    tmp = np.where(~np.isnan(recording))[-1]
    sum_segments = len(tmp)
    sum_of_segments_list.append(sum_segments)

sum_of_valid_segments = []
for recording in real_pairs_segments_removed:
    valid_segments = np.count_nonzero(~np.isnan(recording))
    sum_of_valid_segments.append(valid_segments)

ratios_of_valid_segments = []
for a, b in zip(sum_of_valid_segments, sum_of_segments_list):
    if a == 0 or b == 0:
        ratio = np.nan
    else:
        ratio = a/b * 100
    ratios_of_valid_segments.append(ratio)

ratios_of_valid_segments = np.asarray(ratios_of_valid_segments)

# plt.hist(ratios_of_valid_segments)
# plt.show()

real_pairs_sessions_removed = np.copy(real_pairs_segments_removed)
count_of_sessions_removed = 0
for idx, recording in enumerate(real_pairs_sessions_removed):
    if ratios_of_valid_segments[idx] <= cutoff_secondary:
        real_pairs_sessions_removed[idx, :] = np.nan
        count_of_sessions_removed = count_of_sessions_removed + 1


# for pseudo pairs
sum_of_segments_pseudo = []
for recording in pseudo_pairs:
    tmp = np.where(~np.isnan(recording))[-1]
    sum_segments = len(tmp)
    sum_of_segments_pseudo.append(sum_segments)

sum_of_valid_segments_pseudo = []
for recording in pseudo_pairs_segments_removed:
    valid_segments = np.count_nonzero(~np.isnan(recording))
    sum_of_valid_segments_pseudo.append(valid_segments)

ratios_of_valid_segments_pseudo = []
for a, b in zip(sum_of_valid_segments_pseudo, sum_of_segments_pseudo):
    if a == 0 or b == 0:
        ratio = np.nan
    else:
        ratio = a/b * 100
    ratios_of_valid_segments_pseudo.append(ratio)

ratios_of_valid_segments_pseudo = np.asarray(ratios_of_valid_segments_pseudo)

pseudo_pairs_sessions_removed = np.copy(pseudo_pairs_segments_removed)
count_of_sessions_removed_pseudo = 0
for idx, recording in enumerate(pseudo_pairs_sessions_removed):
    if ratios_of_valid_segments_pseudo[idx] <= cutoff_secondary:
        pseudo_pairs_sessions_removed[idx, :] = np.nan
        count_of_sessions_removed_pseudo = count_of_sessions_removed_pseudo + 1


# let's see how much we cut
valid_segmentsNo = np.count_nonzero(~np.isnan(real_pairs))
valid_segmentsNo_after_filter = np.count_nonzero(~np.isnan(real_pairs_segments_removed))
no_of_segments_removed = valid_segmentsNo - valid_segmentsNo_after_filter
removed_ratio = no_of_segments_removed / valid_segmentsNo * 100

number_of_all_sessions = np.count_nonzero(~np.isnan(real_pairs[:, 0]))
ratio_of_removed_sessions = count_of_sessions_removed / number_of_all_sessions * 100


print('\nFiltering with cutoff: {} removed {}% of segments (real pairs).' .format(cutoff_primary,
                                                                                    np.round(removed_ratio,
                                                                                    decimals=3)))

print('Second filtering with cutoff: {} removed {}% of recordings (real pairs).\n' .format(cutoff_secondary,
                                                                                    np.round(ratio_of_removed_sessions,
                                                                                    decimals=3)))


# get the mean for every remaining session
pseudo_pairs_final = np.nanmean(pseudo_pairs_sessions_removed, axis=1)
real_pairs_final = np.nanmean(real_pairs_sessions_removed, axis=1)

mean_real = np.nanmean(real_pairs_final)
mean_pseudo = np.nanmean(pseudo_pairs_final)
print('\nReal pairs mean dtw: {}' .format(mean_real))
print('Pseudo pairs mean dtw: {}\n' .format(mean_pseudo))


# Plotting

plt.hist(pseudo_pairs_final, bins=50, color='cornflowerblue', label="Pseudo Pairs", alpha=0.5, log=True, density=True)
plt.hist(real_pairs_final, bins=50, color='green', label="Real Pairs", alpha=0.5, log=True, density=True)
plt.axvline(np.nanmean(pseudo_pairs_final), color='mediumblue', linestyle='dashed', label='Mean dist. (pseudo pairs)')
plt.axvline(np.nanmean(real_pairs_final), color='darkgreen', linestyle='dashed', label='Mean dist. (real pairs)')
# plt.yticks([])  # remove tick labels
if cond == 'alap':
    plt.title('Pupil Synchrony for BG, baseline\n', loc='left')
if cond == 'ismeretlen':
    plt.title('Pupil Synchrony for BG, unfamiliar\n', loc='left')
if cond == 'unimodális':
    plt.title('Pupil Synchrony for BG, unimodal\n', loc='left')
if cond == None:
    plt.title('Pupil Synchrony for BG, all\n', loc='left')
plt.xlabel('DTW distance')
plt.ylabel('Density')
plt.legend()
plt.plot()
#gs = gridspec.GridSpec(2, 1)


# Run t-test
ttest_result = scipy.stats.ttest_ind(real_pairs_final, pseudo_pairs_final, equal_var=False, nan_policy='omit')
print(ttest_result)

# Cohen D
def calc_cohenD(array_one, array_two):
    n1, n2 = len(array_one), len(array_two)
    var1, var2 = np.nanvar(array_one, ddof=1), np.nanvar(array_two, ddof=1)
    print('\nVariances: {} {}' .format(var1, var2))
    pooled_std = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    m1, m2 = np.nanmean(array_one), np.nanmean(array_two)
    effect_size = (m1 - m2) / pooled_std
    print("\nCohen's d: {}" .format(effect_size))

    return effect_size

D = calc_cohenD(real_pairs_final, pseudo_pairs_final)

plt.show()

result = [ttest_result, D]


