"""
Script used for creating pseudo pairs for testing pupillary synchrony.

- all possible combinations of pairs per every session (Bargaing Game)
- the script does not filter 'blacklisted' pairs/files

Output: separate numpy arrays for every condition containing BG number, pair number 1 and pair number 2.
"""

import numpy as np
import pandas as pd

all_pairs = (67, 177)
cond = 'ismeretlen'

# extract pairs based on condition
cond_csv_file = "/home/lucab/Documents/commgame_conditions.csv"
cond_df = pd.read_csv(cond_csv_file)
tmp = cond_df.loc[cond_df['Condition'] == cond]
cond_extracted = tmp.loc[tmp['Pair'].between(all_pairs[0], all_pairs[1])]
pairs = np.array(np.unique(cond_extracted['Pair']))
print(pairs)

# get info about how many games per pair are available
BGnos_csv_file = "/home/lucab/Downloads/commGame_pupil_BG_numbers.csv"
tmp_df = pd.read_csv(BGnos_csv_file)
BGnos_df = tmp_df.loc[tmp_df['Pair'].isin(pairs)]

df_BG1 = BGnos_df.loc[BGnos_df["BG1"] != 'not available', ['Pair', 'Labname', 'BG1']]
BG1_pairs = df_BG1['Pair'].to_numpy()
BG1_pairs = np.unique(BG1_pairs)

df_BG2 = BGnos_df.loc[BGnos_df["BG2"] != 'not available', ['Pair', 'Labname', 'BG2']]
BG2_pairs = df_BG2['Pair'].to_numpy()
BG2_pairs = np.unique(BG2_pairs)

df_BG3 = BGnos_df.loc[BGnos_df["BG3"] != 'not available', ['Pair', 'Labname', 'BG3']]
BG3_pairs = df_BG3['Pair'].to_numpy()
BG3_pairs = np.unique(BG3_pairs)

df_BG4 = BGnos_df.loc[BGnos_df["BG4"] != 'not available', ['Pair', 'Labname', 'BG4']]
BG4_pairs = df_BG4['Pair'].to_numpy()
BG4_pairs = np.unique(BG4_pairs)


# BG1
BG1_pseudo_pairs = np.zeros((len(BG1_pairs)*len(BG1_pairs), 3))
counter = 0
for m in BG1_pairs:
    for g in BG1_pairs:
        BG1_pseudo_pairs[counter, 0] = 1
        BG1_pseudo_pairs[counter, 1] = m
        BG1_pseudo_pairs[counter, 2] = g
        counter += 1
BG1_pseudo_pairs = BG1_pseudo_pairs[BG1_pseudo_pairs[:, 1] != BG1_pseudo_pairs[:, 2]]

print(np.shape(BG1_pseudo_pairs))


# BG2
BG2_pseudo_pairs = np.zeros((len(BG2_pairs)*len(BG2_pairs), 3))
counter = 0
for m in BG2_pairs:
    for g in BG2_pairs:
        BG2_pseudo_pairs[counter, 0] = 2
        BG2_pseudo_pairs[counter, 1] = m
        BG2_pseudo_pairs[counter, 2] = g
        counter += 1
BG2_pseudo_pairs = BG2_pseudo_pairs[BG2_pseudo_pairs[:, 1] != BG2_pseudo_pairs[:, 2]]


# BG3
BG3_pseudo_pairs = np.zeros((len(BG3_pairs)*len(BG3_pairs), 3))
counter = 0
for m in BG3_pairs:
    for g in BG3_pairs:
        BG3_pseudo_pairs[counter, 0] = 3
        BG3_pseudo_pairs[counter, 1] = m
        BG3_pseudo_pairs[counter, 2] = g
        counter += 1
BG3_pseudo_pairs = BG3_pseudo_pairs[BG3_pseudo_pairs[:, 1] != BG3_pseudo_pairs[:, 2]]


# BG4
BG4_pseudo_pairs = np.zeros((len(BG4_pairs)*len(BG4_pairs), 3))
counter = 0
for m in BG4_pairs:
    for g in BG4_pairs:
        BG4_pseudo_pairs[counter, 0] = 4
        BG4_pseudo_pairs[counter, 1] = m
        BG4_pseudo_pairs[counter, 2] = g
        counter += 1

BG4_pseudo_pairs = BG4_pseudo_pairs[BG4_pseudo_pairs[:, 1] != BG4_pseudo_pairs[:, 2]]


all_pseudo_pairs = np.concatenate((BG1_pseudo_pairs, BG2_pseudo_pairs, BG3_pseudo_pairs, BG4_pseudo_pairs), axis=0)

if cond == 'alap':
    np.save("/media/lucab/data_hdd/pseudo_pairs_list_new_fam", all_pseudo_pairs)
if cond == 'ismeretlen':
    np.save("/media/lucab/data_hdd/pseudo_pairs_list_new_unfam", all_pseudo_pairs)
if cond == 'unimodális':
    np.save("/media/lucab/data_hdd/pseudo_pairs_list_new_uni", all_pseudo_pairs)

