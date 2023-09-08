# pupil_synchrony

Scripts used for calculating synchrony between pairs of participants with Dyanmic Time Warping (DTW).

How to use:

1) create_pseudo_pairs.py

Run this first. You will need the commgame_conditions.csv and the commGame_pupil_BG_numbers.csv files. The code generates pseudo pairs (in reality pseudo 'sessions') for every condition.

2) pupil_dtw_sync_v2.py 

This script creates windows of the time series data and calculates DTW for every window (default parameters: 6 sec window length with 3 sec step size). 
No filtering based on NaN values at this stage. Output is saved in numpy.
Hardcoded inputs: root directory, range of pair numbers, session names, condition and names of output files.

Note: a blacklist is included for both dtw scripts, but not actually used here (no resampled data were created for blakclisted files). 
 
3) pupil_dtw_sync_pseudo.py

Same logic as pupil_dtw_sync_v2.py, but tailored to pseudo pairs. Output is saved in numpy. 
Hardcoded inputs: root directory, condition and names of output files.

4) t_test.py

Run this last. The code filters the DTW result based on missing values (NaN-s), plots histogramms for real vs pseudo pairs and calculates statistical difference. 
Hardcoded inputs: condition, primary cutoff value, secondary cutoff value and the names of the result files generated from step 2) and 3).


