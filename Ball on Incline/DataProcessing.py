import pandas as pd
import numpy as np
import os

def weighted_error_prop_mean(data_frame_column_variable, data_frame_column_sig):
    mean, inverse_sigsum_sq = 0, 0
    sig_list = data_frame_column_sig.dropna().tolist()
    var_list = data_frame_column_variable.dropna().tolist()
    for j in range(len(sig_list)):
        inverse_sigsum_sq += (1/(sig_list[j]**2))
    for i in range(len(var_list)):
        mean += (var_list[i]/(sig_list[i]**2)) / inverse_sigsum_sq
    uncert = np.sqrt(1/inverse_sigsum_sq)
    print("The mean of", data_frame_column_variable.name, "is: ", f"{mean:.2f}", " and the uncertainty is: +-", f"{uncert:.2f}")
    return mean, uncert


file_path = str(os.getcwd() + r"/BallInclineData.csv")


df = pd.read_csv(file_path)

L1_mean, L1_uncert = weighted_error_prop_mean(df['L_1 (mm)'], df['L_1_sig (mm)'])
L2_mean, L2_uncert = weighted_error_prop_mean(df['L_2 (mm)'], df['L_2_sig (mm)'])
L3_mean, L3_uncert = weighted_error_prop_mean(df['L_3 (mm)'], df['L_3_sig (mm)'])
L4_mean, L4_uncert = weighted_error_prop_mean(df['L_4 (mm)'], df['L_4_sig (mm)'])
L5_mean, L5_uncert = weighted_error_prop_mean(df['L_5 (mm)'], df['L_5_sig (mm)'])
r_bball_mean, r_bball_uncert = weighted_error_prop_mean(df['r_bball (mm)'], df['r_bball_sig (mm)'])
theta_mean, theta_uncert = weighted_error_prop_mean(df['theta (degrees)'], df['theta_sig (degrees)'])
