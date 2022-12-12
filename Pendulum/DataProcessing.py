import os
import sympy
from sympy import *
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
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


def length_addition(x_val, x_val_sig, y_val, y_val_sig):              # Assuming the period and length are uncorrelated
    x, y = symbols("x, y")                                                              # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                                                # Assigning symbolic nature to variables
    L = x + (1/2)*y                                                                     # Rope + 1/2 weight, since center of mass
    dL = sqrt((L.diff(x)*dx)**2 + (L.diff(y)*dy)**2)                                    # The function for the error propagation
    fL = lambdify((x,y), L)                                                             # Make it a numerical function
    fdL = lambdify((x, dx, y, dy), dL)                                                  # Make it a numerical function
    vLength, vdLength = x_val, x_val_sig                                              # Insert values for first parameter
    vWeight_length, vdWeight_length = y_val, y_val_sig                  # Insert values for second parameter
    vL = fL(vLength, vWeight_length)                                                    # Calculate the numerical function
    vdL = fdL(vLength, vdLength, vWeight_length, vdWeight_length)                       # Calculate the numerical function of the uncertainty
    #print("The calculated length with propagated errors is: " f"{vL:.2f}", "+-", f"{vdL:.2f}")
    return vL, vdL

def addition(x_val, x_val_sig, y_val, y_val_sig):              # Assuming the period and length are uncorrelated
    x, y = symbols("x, y")                                                              # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                                                # Assigning symbolic nature to variables
    L = (x + y)/2                                                                     # Slightly uncertain about how to handle this
    dL = sqrt((L.diff(x)*dx)**2 + (L.diff(y)*dy)**2)                                    # The function for the error propagation
    fL = lambdify((x,y), L)                                                             # Make it a numerical function
    fdL = lambdify((x, dx, y, dy), dL)                                                  # Make it a numerical function
    vLength, vdLength = x_val, x_val_sig                                              # Insert values for first parameter
    vWeight_length, vdWeight_length = y_val, y_val_sig                  # Insert values for second parameter
    vL = fL(vLength, vWeight_length)                                                    # Calculate the numerical function
    vdL = fdL(vLength, vdLength, vWeight_length, vdWeight_length)                       # Calculate the numerical function of the uncertainty
    #print("The calculated length with propagated errors is: " f"{vL:.2f}", "+-", f"{vdL:.2f}")
    return vL, vdL

# Define a symbolic function for g, to calculate error prop in symbolic functions

def g_Pendulum(T, T_sig, L, L_sig):                                 # Assuming the period and length are uncorrelated
    L_sym, T_sym = symbols("l, t")                                          # Assigning symbolic nature to variables
    dL, dT = symbols("sigma_L, sigma_T")                            # Assigning symbolic nature to variables

    g = L_sym*(2*np.pi/T_sym)**2                                            # Defining the function for the expression    
    dg = sqrt((g.diff(L_sym)*dL)**2 + (g.diff(T_sym)*dT)**2)                # The function for the error propagation

    fg = lambdify((L_sym, T_sym), g)                                         # Make it a numerical function
    fdg = lambdify((L_sym, dL, T_sym, dT), dg)                              # Make it a numerical function
    
    vL, vdL = L, L_sig                                              # Insert values for first parameter
    vT, vdT = T, T_sig                                              # Insert values for second parameter
    vg = fg(vL, vT)                                                 # Calculate the numerical function
    vdg = fdg(vL, vdL, vT, vdT)                                     # Calculate the numerical function of the uncertainty
    print("The calculated gravitional acceleration with propagated errors is: " f"{vg:.2f}", "+-", f"{vdg:.2f} using {L:.2f} as the length")
    return vg, vdg

# Start working on the data-processing of the timers of the pendulum

# Set data paths
# %% Calculate periods
def load_times():

    # Generate a list of .dat files
    data_files = []
    files = os.listdir(os.getcwd())
    for file in files:
        if file[-3:] == 'dat':
            data_files.append(file)

    # Load files into dataframe
    df = pd.DataFrame(columns=['Period', 'Time', 'Filename'])
    df_t = df.copy()
    for file in data_files:
        df_t = pd.DataFrame(columns=['Period', 'Time', 'Filename'])

        # Load data from file
        arr = np.loadtxt(file)
        df_t[['Period', 'Time']] = arr
        df_t['Filename'] = file

        # if file == 'Pendulum_Klas_good_26_and_last_are_bad.dat':
        #     return df_t
            
        # Save into dataframe
        if len(df) == 0:
            df = df_t.copy()
        else:
            df = df.append(df_t, ignore_index=True).reset_index(drop=True)
    return df

df = load_times()

# %% Calculate periods
from iminuit import Minuit                             # The actual fitting tool, better than scipy's
from ExternalFunctions import Chi2Regression, BinnedLH, UnbinnedLH
from ExternalFunctions import nice_string_output, add_text_to_ax    # Useful functions to print fit results on figure

def poly_1_deg(x, slope, offset):
    return slope*x + offset

df = load_times()
markers = ['x', '^', 'v']
markers = ['o']*3
files = df['Filename'].unique()
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=[14, 7.2/2], sharex=True, sharey=True)
size = 5

names = ['Alexander', 'Klas', 'Arnulf']
for i in range(len(files)):
    print('-'*10)
    ax = axes[i]
    axt = plt.twinx(ax)
    axins = ax.inset_axes([20, 50, 16, 100], transform=ax.transData)
    ax.set_title(names[i])
    file = files[i]
    ind = df['Filename'] == file
    
    x = df.loc[ind, 'Period'].values
    y = df.loc[ind, 'Time'].values


    # First we do a least square fit
    p = np.polyfit(x, y, 1)
    errors = np.std(y - np.polyval(p, x))

    # Remove dead time before start point
    # x = x - np.min(x)
    # y = y - np.min(y)


    chi2_object = Chi2Regression(poly_1_deg, x, y, np.ones(len(x))*errors)
    minuit_pendelum = Minuit(chi2_object, slope=8.0, offset=0)
    minuit_pendelum.errordef = 1.0   # Chi2 fit
    minuit_pendelum.migrad();
    p = minuit_pendelum.values[:]

    print(names[i], p)
    print(p)
    # Calculate errors - Should it be squared?
    yerr = (y - poly_1_deg(x, *p))

    df.loc[ind, 'Residuals'] = yerr
    df.loc[ind, 'Period_time'] = df.loc[ind, 'Time'].diff()
    
    # Add all measurements together
    if i == 0:
        X = x
        Y = y
        Ye = yerr
    else:
        X = np.concatenate([X, x])
        Y = np.concatenate([Y, y])
        Ye = np.concatenate([Ye, yerr])

    ax.errorbar(x, y, yerr=yerr, ls='', marker='o', ms=4)
    axt.errorbar(x, yerr, yerr=np.std(yerr), marker=markers[i], ms=2, ls='')
    ax.plot(x, poly_1_deg(x, *p), 'r-')
    hist, bins = np.histogram(yerr, bins=np.linspace(-0.3, 0.3, 13))
    bin_centers = bins[:-1] + np.mean(np.diff(bins))/2
    axins.errorbar(bin_centers, hist, yerr=hist/np.sqrt(hist),
                   xerr=np.diff(bins)[-1], ls='', lw=0.5)

    # Fit Gaussian
    def gaussian(x, a, mu, sigma):
        return a*np.exp(-((x - mu)/sigma)**2)

    chi2_object_errors = Chi2Regression(gaussian, bin_centers, hist, hist/np.sqrt(hist))
    minuit_error_fit = Minuit(chi2_object_errors , a=15, mu=0, sigma=0.1)
    minuit_error_fit.errordef = 1.0   # Chi2 fit
    minuit_error_fit.migrad();
    p = minuit_error_fit.values[:]
    axins.plot(np.linspace(-0.3, 0.3, 100), gaussian(np.linspace(-0.3, 0.3, 100), *p))

    # Get the ChiSquare probability:
    chi2_lin = minuit_pendelum.fval
    ndof_lin = len(x) - len(minuit_pendelum.values[:])
    chi2_prob_lin = stats.chi2.sf(chi2_lin, ndof_lin)
    
    # Include fit results in the plot:
    d = {'Chi2': chi2_lin,
         'Ndof': ndof_lin,
         'Prob': chi2_prob_lin,
         'slope': [minuit_pendelum.values['slope'], minuit_pendelum.errors['slope']],
         'offset': [minuit_pendelum.values['offset'], minuit_pendelum.errors['offset']],
        }
    
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.02, 0.97, text, ax, fontsize=8);

    # Make everything look nice
    # ax.set_ylim([-30, 1.1*y[-1]])
    ax.set_ylim([-30, 365])
    yticks = np.linspace(-0.3, 0.3, 5)
    axt.set_ylim(np.array(ax.get_ylim())/100)
    axt.set_yticks(yticks)
    ax.grid()
    
    ax.set_xlim(x[0]-1, x[-1]+1)
    
    # Standard deviation of residuals
    std_err = np.std(Ye)
    axt.plot(ax.get_xlim(), [std_err, std_err], c='grey', ls=':')
    axt.plot(ax.get_xlim(), [-std_err, -std_err], c='grey', ls=':')
    axt.set_yticklabels(np.round(yticks, 2), size='x-small',
                        color='blue')  # ????
    
    # Small size
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    
    # Small size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('small')
    
    ax.text(1.08, 0.25, 'Dashed lines show ' + r'$\pm 1\sigma$', rotation=90,
            size='xx-small', transform=ax.transAxes, va='bottom')
    # Labels
    ax.set_ylabel('Time (s)', size='small', labelpad=0.5)
    ax.set_xlabel('Measurement(#)', size='small', labelpad=0.5)
    axt.set_ylabel('Time residuals (s)                   ', size='small',
                   loc='center', labelpad=-20, color='blue')


    # Actually quantify the period
    slope = minuit_pendelum.values['slope']
    slope_error = minuit_pendelum.errors['slope']

    # L_mean, L_mean_sig = weighted_error_prop_mean(Length, Length_sig, "L")

    # print('Gravity:', g_Pendulum(slope, slope_error, L_mean, L_mean_sig))

fig.subplots_adjust(top=0.9, right=0.9, left=0.15)
fig.savefig('Pendelum.png', dpi=300)
# %%
path_to_timer_dat_arnulf = str(os.getcwd() + r"/periodmeasure2_arnulf.dat")
path_to_timer_dat_alex = str(os.getcwd() + r"/alex_output_1.dat")
path_to_data = str(os.getcwd() + r"/PendulumData.csv")
path_to_T_raw_sig = str(os.getcwd() + r"/T_raw_sig.csv")

file_alex = np.loadtxt(path_to_timer_dat_alex)                                                    # Load the .dat file
array_of_times_alex = np.zeros(len(file_alex))                                                    # Set a clean numpy array to load the data into
for i in range(len(file_alex)):                                                                   # Iterate through the .dat file
    array_of_times_alex[i] = float(list(str(file_alex[i][:]).split(".   ", 1))[1].split("]")[0])  # This handles the weird format into clean floats

file_arnulf = np.loadtxt(path_to_timer_dat_arnulf)                                                    # Load the .dat file
array_of_times_arnulf = np.zeros(len(file_arnulf))                                                    # Set a clean numpy array to load the data into
for i in range(len(file_arnulf)):                                                                   # Iterate through the .dat file
    array_of_times_arnulf[i] = float(list(str(file_arnulf[i][:]).split(".   ", 1))[1].split("]")[0])  # This handles the weird format into clean floats


df = pd.read_csv(path_to_data) 
df_T_raw_sig = pd.read_csv(path_to_T_raw_sig)                                                             # Pandas to read the .csv file.

#for i in range(len(df.columns)):
#    df.iloc[i].dropna()

T_raw_sig = df_T_raw_sig["T_raw_sig (s)"].tolist()                                                    # Pandas read sigs from data file

array_of_times = np.zeros(len(array_of_times_alex))
array_of_times_sig = np.zeros(len(array_of_times_alex))

T_alex = np.zeros(len(array_of_times_alex) - 1)                                                       # Initialise period array
T_alex_sig = np.zeros(len(array_of_times_alex) - 1)                                                   # Initialise period_sig array
for i in range(1, len(array_of_times_alex)):                                                     # Iterate to calculate T and T_sig, using previous function
    T_alex[i - 1], T_alex_sig[i - 1] = period_Pendulum(array_of_times_alex[i - 1], T_raw_sig[i-1], array_of_times_alex[i], T_raw_sig[i])

T_arnulf = np.zeros(len(array_of_times_arnulf) - 1)                                                       # Initialise period array
T_arnulf_sig = np.zeros(len(array_of_times_arnulf) - 1)                                                   # Initialise period_sig array
for i in range(1, len(array_of_times_arnulf)):                                                     # Iterate to calculate T and T_sig, using previous function
    T_arnulf[i - 1], T_arnulf_sig[i - 1] = period_Pendulum(array_of_times_arnulf[i - 1], T_raw_sig[i-1], array_of_times_arnulf[i], T_raw_sig[i])


T = np.zeros(len(T_alex))
T_sig = np.zeros(len(T_alex))
for i in range(len(T_alex)):
    T[i], T_sig[i] = addition(T_alex[i], T_alex_sig[i], T_arnulf[i], T_arnulf_sig[i])

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


# %%
# Assuming the period and length are uncorrelated

g_Pendulum(T, T_sig, L, L_sig)
