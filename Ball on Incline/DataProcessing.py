import pandas as pd
import numpy as np
import sympy as sp
from sympy import *
import os

# Define function to calculate the error propagation in collected data using a weighted mean ------------------------------------------------

def weighted_error_prop_mean(data_frame_column_variable, data_frame_column_sig): # Using data_frames and specific columns
    mean, inverse_sigsum_sq = 0, 0                                               # Initialise values
    sig_list = data_frame_column_sig.dropna().tolist()                           # Convert data column to list, excluding all NaN values
    var_list = data_frame_column_variable.dropna().tolist()                      # Convert data column to list, excluding all NaN values
    for j in range(len(sig_list)):                                               # Calculate inverse sum of squared sig; normalise expression
        inverse_sigsum_sq += (1/(sig_list[j]**2))
    for i in range(len(var_list)):                                               # Weighted mean: equation from slide 16 of "Mean and Width"
        mean += (var_list[i]/(sig_list[i]**2)) / inverse_sigsum_sq
    uncert = np.sqrt(1/inverse_sigsum_sq)                                        # Uncertainty from slide 16 of "Mean and Width"
    print("The mean of", data_frame_column_variable.name, "is: ", f"{mean:.2f}", " and the uncertainty is: +-", f"{uncert:.2f}")
    return mean, uncert


file_path = str(os.getcwd() + r"/BallInclineData.csv")                   # Set path to find the file; should work with folder synced with Git


df = pd.read_csv(file_path)                                             # Pandas to read the .csv file.

L1_mean, L1_uncert = weighted_error_prop_mean(df['L_1 (mm)'], df['L_1_sig (mm)'])
L2_mean, L2_uncert = weighted_error_prop_mean(df['L_2 (mm)'], df['L_2_sig (mm)'])
L3_mean, L3_uncert = weighted_error_prop_mean(df['L_3 (mm)'], df['L_3_sig (mm)'])
L4_mean, L4_uncert = weighted_error_prop_mean(df['L_4 (mm)'], df['L_4_sig (mm)'])
L5_mean, L5_uncert = weighted_error_prop_mean(df['L_5 (mm)'], df['L_5_sig (mm)'])
r_bball_mean, r_bball_uncert = weighted_error_prop_mean(df['r_bball (mm)'], df['r_bball_sig (mm)'])
r_sball_mean, r_sball_uncert = weighted_error_prop_mean(df['r_sball (mm)'], df['r_sball_sig (mm)'])
theta_mean, theta_uncert = weighted_error_prop_mean(df['theta (degrees)'], df['theta_sig (degrees)'])
L_mean, L_uncert = weighted_error_prop_mean(df['L (mm)'], df['L_sig (mm)'])
Hyp_mean, Hyp_uncert = weighted_error_prop_mean(df['Hyp (mm)'], df['Hyp_sig (mm)'])
h_mean, h_uncert = weighted_error_prop_mean(df['h (mm)'], df['h_sig (mm)'])


def angle(Hyp, Opp, Hyp_uncern, Opp_uncern):                        # Assuming the lengths are not correlated
    x, y = symbols("x, y")
    dx, dy = symbols("sigma_x, sigma_y")
    Theta = atan(x/y)
    dTheta = sqrt((Theta.diff(x)*dx)**2 + (Theta.diff(y)*dy)**2)
    fTheta = lambdify((x,y), Theta)
    fdTheta = lambdify((x, dx, y, dy), dTheta)
    vx, vdx = Opp, Opp_uncern
    vy, vdy = Hyp, Hyp_uncern
    vTheta = fTheta(vx, vy)
    vdTheta = fdTheta(vx, vdx, vy, vdy)
    print("The calculated angle with propagated errors is: " f"{vTheta:.2f}", "+-", f"{vdTheta:.2f}")
    return vTheta, vdTheta

theta = angle(L_mean, h_mean, L_uncert, h_uncert)