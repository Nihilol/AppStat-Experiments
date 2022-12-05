import os
import sympy
from sympy import *
import numpy as np
import pandas as pd
import sys

# Define functions used in the data processing: ------------------------------------------------------------------------------------------------

# Define a function for weighted error propagation using dataframe columns

def weighted_error_prop_mean_df(data_frame_column_variable, data_frame_column_sig): # Using data_frames and specific columns
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

# Define a function for weigthed error prop using lists/arrays

def weighted_error_prop_mean(var_list, sig_list, name):                          # Using data_frames and specific columns
    mean, inverse_sigsum_sq = 0, 0                                               # Initialise values
    for j in range(len(sig_list)):                                               # Calculate inverse sum of squared sig; normalise expression
        inverse_sigsum_sq += (1/(sig_list[j]**2))
    for i in range(len(var_list)):                                               # Weighted mean: equation from slide 16 of "Mean and Width"
        mean += (var_list[i]/(sig_list[i]**2)) / inverse_sigsum_sq
    uncert = np.sqrt(1/inverse_sigsum_sq)                                        # Uncertainty from slide 16 of "Mean and Width"
    print("The mean of the", name, "is: ", f"{mean:.2f}", " and the uncertainty is: +-", f"{uncert:.2f}")
    return mean, uncert


# Define a symbolic function for calculation the period with Sympy, which helps calculate error propagation in symbolic expressions

def period_Pendulum(val_prev, val_prev_sig, val_now, val_now_sig):                      # Assuming the period and length are uncorrelated
    x, y = symbols("x, y")                                                              # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                                                # Assigning symbolic nature to variables
    T = x - y                                                                           # Defining the function for the expression
    dT = sqrt((T.diff(x)*dx)**2 + (T.diff(y)*dy)**2)                                    # The function for the error propagation
    fT = lambdify((x,y), T)                                                             # Make it a numerical function
    fdT = lambdify((x, dx, y, dy), dT)                                                  # Make it a numerical function
    vval_now, vdval_now = val_now, val_now_sig                                          # Insert values for first parameter
    vval_prev, vdval_prev = val_prev, val_prev_sig                                      # Insert values for second parameter
    vT = fT(vval_now, vval_prev)                                                        # Calculate the numerical function
    vdT = fdT(vval_now, vdval_now, vval_prev, vdval_prev)                               # Calculate the numerical function of the uncertainty
    #print("The calculated period with propagated errors is: " f"{vT:.2f}", "+-", f"{vdT:.2f}")
    return vT, vdT


def length_addition(Length, Length_sig, Weight_length, Weight_length_sig):              # Assuming the period and length are uncorrelated
    x, y = symbols("x, y")                                                              # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                                                # Assigning symbolic nature to variables
    L = x + (1/2)*y                                                                     # Rope + 1/2 weight, since center of mass
    dL = sqrt((L.diff(x)*dx)**2 + (L.diff(y)*dy)**2)                                    # The function for the error propagation
    fL = lambdify((x,y), L)                                                             # Make it a numerical function
    fdL = lambdify((x, dx, y, dy), dL)                                                  # Make it a numerical function
    vLength, vdLength = Length, Length_sig                                              # Insert values for first parameter
    vWeight_length, vdWeight_length = Weight_length, Weight_length_sig                  # Insert values for second parameter
    vL = fL(vLength, vWeight_length)                                                    # Calculate the numerical function
    vdL = fdL(vLength, vdLength, vWeight_length, vdWeight_length)                       # Calculate the numerical function of the uncertainty
    #print("The calculated length with propagated errors is: " f"{vL:.2f}", "+-", f"{vdL:.2f}")
    return vL, vdL

# Define a symbolic function for g, to calculate error prop in symbolic functions

def g_Pendulum(T, T_sig, L, L_sig):                                 # Assuming the period and length are uncorrelated
    x, y = symbols("x, y")                                          # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                            # Assigning symbolic nature to variables
    g = x*(2*np.pi/y)**2                                            # Defining the function for the expression
    dg = sqrt((g.diff(x)*dx)**2 + (g.diff(y)*dy)**2)                # The function for the error propagation
    fg = lambdify((x,y), g)                                         # Make it a numerical function
    fdg = lambdify((x, dx, y, dy), dg)                              # Make it a numerical function
    vL, vdL = L, L_sig                                              # Insert values for first parameter
    vT, vdT = T, T_sig                                              # Insert values for second parameter
    vg = fg(vL, vT)                                                 # Calculate the numerical function
    vdg = fdg(vL, vdL, vT, vdT)                                     # Calculate the numerical function of the uncertainty
    print("The calculated gravitional velocity with propagated errors is: " f"{vg:.2f}", "+-", f"{vdg:.2f} using {L:.2f} as the length")
    return vg, vdg

# Start working on the data-processing of the timers of the pendulum

# Set data paths

path_to_timer_dat = str(os.getcwd() + r"/alex_output_1.dat")
path_to_data = str(os.getcwd() + r"/PendulumData.csv")
path_to_T_raw_sig = str(os.getcwd() + r"/T_raw_sig.csv")

file = np.loadtxt(path_to_timer_dat)                                                        # Load the .dat file
array_of_times = np.zeros(len(file))                                                        # Set a clean numpy array to load the data into
for i in range(len(file)):                                                                  # Iterate through the .dat file
    array_of_times[i] = float(list(str(file[i][:]).split(".   ", 1))[1].split("]")[0])      # This handles the weird format into clean floats

df = pd.read_csv(path_to_data) 
df_T_raw_sig = pd.read_csv(path_to_T_raw_sig)                                                             # Pandas to read the .csv file.

#for i in range(len(df.columns)):
#    df.iloc[i].dropna()

T_raw_sig = df_T_raw_sig["T_raw_sig (s)"].tolist()                                                    # Pandas read sigs from data file

T = np.zeros(len(array_of_times) - 1)                                                       # Initialise period array
T_sig = np.zeros(len(array_of_times) - 1)                                                   # Initialise period_sig array
for i in range(1, len(array_of_times)):                                                     # Iterate to calculate T and T_sig, using previous function
    T[i - 1], T_sig[i - 1] = period_Pendulum(array_of_times[i - 1], T_raw_sig[i-1], array_of_times[i], T_raw_sig[i])

Length_las = np.zeros(len(df["w_h (m)"]))                                                   # Initialise laser length array
Length_las_sig = np.zeros(len(df["w_h (m)"]))                                               # Initialise laser length sigs
Length = np.zeros(len(df["w_h (m)"]))                                                       # Initialise length array
Length_sig = np.zeros(len(df["w_h (m)"]))                                                   # Initialise length sigs
for i in range(len(df["w_h (m)"])):                                                         # Use previous symbolic function to calculate length
    Length_las[i], Length_las_sig[i] = length_addition(df["Len_las (m)"][i], df["Len_las_sig (m)"][i], df["w_h (m)"][i], df["w_h_sig (m)"][i])
    Length[i], Length_sig[i] = length_addition(df["Len (m)"][i], df["Len_sig (m)"][i], df["w_h (m)"][i], df["w_h_sig (m)"][i])


T_mean, T_sig_mean = weighted_error_prop_mean(T, T_sig, "period")                                    # Combine values to get weighted mean and error prop of T
L_las_mean, L_las_mean_sig = weighted_error_prop_mean(Length_las, Length_las_sig, "L_laser")
L_mean, L_mean_sig = weighted_error_prop_mean(Length, Length_sig, "L")
g_las = g_Pendulum(T_mean, T_sig_mean, L_las_mean, L_las_mean_sig)
g = g_Pendulum(T_mean, T_sig_mean, L_mean, L_mean_sig)
#print("The value of the gravitional acceleration, with no error propagation, is: "f"{g:.3f}")
