import numpy as np
import pandas as pd
import os

def weight_uncert(inverse_sigs):
    uncert = np.sqrt(1/inverse_sum_sigs_bball)
    return uncert

file_path = str(os.getcwd() + r"/BallInclineData.csv")


df = pd.read_csv(file_path)

r_bball_mean, r_sball_mean, inverse_sum_sigs_bball, inverse_sum_sigs_sball, inverse_sum_sigs_L1, inverse_sum_sigs_L2 = 0, 0, 0, 0, 0, 0
inverse_sum_sigs_L3, inverse_sum_sigs_L4, inverse_sum_sigs_L5, L1_mean, L2_mean, L3_mean, L4_mean, L5_mean = 0, 0, 0, 0, 0, 0, 0, 0

# This section all calculates errors using a weighted mean: ------------------------------------------------------------------------------------------

for i in range(len(df['r_bball_sig (mm)'].tolist())):
    inverse_sum_sigs_bball += (1/(df['r_bball_sig (mm)'].dropna().tolist()[i]**2))
    inverse_sum_sigs_sball += (1/(df['r_sball_sig (mm)'].dropna().tolist()[i]**2))
    try:
        inverse_sum_sigs_L1 += (1/(df['L_1_sig (mm)'].dropna().tolist()[i]**2))
        inverse_sum_sigs_L2 += (1/(df['L_2_sig (mm)'].dropna().tolist()[i]**2))
        inverse_sum_sigs_L3 += (1/(df['L_3_sig (mm)'].dropna().tolist()[i]**2))
        inverse_sum_sigs_L4 += (1/(df['L_4_sig (mm)'].dropna().tolist()[i]**2))
        inverse_sum_sigs_L5 += (1/(df['L_5_sig (mm)'].dropna().tolist()[i]**2))
    except IndexError:
        None


for i in range(len(df['r_bball (mm)'].tolist())):
    r_bball_mean += (df['r_bball (mm)'].dropna().tolist()[i]/(df['r_bball_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_bball
    r_sball_mean += (df['r_sball (mm)'].dropna().tolist()[i]/(df['r_sball_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_sball
    try:
        L1_mean += (df['L_1 (mm)'].dropna().tolist()[i]/(df['L_1_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_L1
        L2_mean += (df['L_2 (mm)'].dropna().tolist()[i]/(df['L_2_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_L2
        L3_mean += (df['L_3 (mm)'].dropna().tolist()[i]/(df['L_3_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_L3
        L4_mean += (df['L_4 (mm)'].dropna().tolist()[i]/(df['L_4_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_L4
        L5_mean += (df['L_5 (mm)'].dropna().tolist()[i]/(df['L_4_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_L5
    except IndexError:
        None


uncert_on_r_bball_mean = weight_uncert(inverse_sum_sigs_bball)
uncert_on_r_sball_mean = weight_uncert(inverse_sum_sigs_sball)
uncert_on_L1_mean = weight_uncert(inverse_sum_sigs_L1)
uncert_on_L2_mean = weight_uncert(inverse_sum_sigs_L2)
uncert_on_L3_mean = weight_uncert(inverse_sum_sigs_L3)
uncert_on_L4_mean = weight_uncert(inverse_sum_sigs_L4)
uncert_on_L5_mean = weight_uncert(inverse_sum_sigs_L5)

print("The radius of the big sphere is: ", f"{r_bball_mean:.2f}", " +- " , f"{uncert_on_r_bball_mean:.2f}", " with propagated errors")
print("The radius of the small sphere is: ", f"{r_sball_mean:.2f}", " +- " , f"{uncert_on_r_sball_mean:.2f}", " with propagated errors")
print("The length of L1 is: ", f"{L1_mean:.2f}", " +- " , f"{uncert_on_r_bball_mean:.2f}", " with propagated errors")
print("The length of L2 is: ", f"{L2_mean:.2f}", " +- " , f"{uncert_on_r_sball_mean:.2f}", " with propagated errors")
print("The length of L3 is: ", f"{L3_mean:.2f}", " +- " , f"{uncert_on_r_bball_mean:.2f}", " with propagated errors")
print("The length of L4 is: ", f"{L4_mean:.2f}", " +- " , f"{uncert_on_r_sball_mean:.2f}", " with propagated errors")
print("The length of L5 is: ", f"{L5_mean:.2f}", " +- " , f"{uncert_on_r_bball_mean:.2f}", " with propagated errors")