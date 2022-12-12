
import os
import sympy         # Whats going on here?
from sympy import *  # And here?
import numpy as np
import pandas as pd
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
theta_rev_mean, theta_rev_uncert = weighted_error_prop_mean(df['theta_rev (degrees)'], df['theta_rev_sig (degrees)'])
L_mean, L_uncert = weighted_error_prop_mean(df['L (mm)'], df['L_sig (mm)'])
Hyp_mean, Hyp_uncert = weighted_error_prop_mean(df['Hyp (mm)'], df['Hyp_sig (mm)'])
h_mean, h_uncert = weighted_error_prop_mean(df['h (mm)'], df['h_sig (mm)'])
d_r_mean, d_r_uncert = weighted_error_prop_mean(df['d_rail (mm)'], df['d_rail_sig (mm)'])

# Define function to calculate the angle of incline ------------------------------------------------------------------------------------------------
def angle(Hyp, Opp, Hyp_uncern, Opp_uncern):                        # Assuming the lengths are not correlated
    x, y = symbols("x, y")                                          # Assigning symbolic nature to variables
    dx, dy = symbols("sigma_x, sigma_y")                            # Assigning symbolic nature to variables
    Theta = asin(x/y)                                               # Defining the function for the angle
    dTheta = sqrt((Theta.diff(x)*dx)**2 + (Theta.diff(y)*dy)**2)    # The function for the error propagation
    fTheta = lambdify((x,y), Theta)                                 # Make it a numerical function
    fdTheta = lambdify((x, dx, y, dy), dTheta)                      # Make it a numerical function
    vx, vdx = Opp, Opp_uncern                                       # Insert values for first parameter
    vy, vdy = Hyp, Hyp_uncern                                       # Insert values for second parameter
    vTheta = fTheta(vx, vy)                                         # Calculate the numerical function
    vdTheta = fdTheta(vx, vdx, vy, vdy)                             # Calculate the numerical function of the uncertainty
    vTheta = (vTheta*180)/np.pi                                     # Convert output to degrees
    vdTheta = (vdTheta*180)/np.pi                                   # Convert output to degrees
    print("The calculated angle with propagated errors is: " f"{vTheta:.2f}", "+-", f"{vdTheta:.2f}")
    return vTheta, vdTheta

theta = angle(Hyp_mean, h_mean, Hyp_uncert, h_uncert)

#We load the time data for Big and small spheres into two arrays, as well as two for the reverse cases

#Since we have an uneven number of data points, we drop the first data points from each array untill they all have length of the shortest
#We can check that we haven't lost any peaks and so there won't be a problem.
# %%
# Simple csv load function
def load_balls(folder):
    signals = []
    files = []

    for file in os.listdir(folder):
        print('Loading', file)
        df = pd.read_csv(folder + file, header=13)
        files.append(file)
        signal = df.values
        signal[:, 0] = signal[:, 0] - signal[0, 0]
        signals.append(df.values)
        
    return signals, files


# Calculate Full Width -Half maximum of peaks
def FWHM(X, Y):
    half_max = max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X[0:len(d)],d) #if you are interested
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0][0]
    right_idx = np.where(d < 0)[0][-1]
    return X[right_idx] - X[left_idx] #return the difference (full width)


# Finds peaks and their widths
def find_peaks(array, plot=True):
    diffs = np.diff(array[:, 1])
    peak_x, _ = ss.find_peaks(diffs, height=np.std(diffs)*2, distance=1000)
    
    if plot:
        fig, ax = plt.subplots(1)
        # Subsampled to save memory...
        ax.plot(array[::10, 0], array[::10, 1])
        ax.plot(array[1:, 0], diffs)
        ax.scatter(array[peak_x, 0], diffs[peak_x], marker='*')

        
    widths = np.zeros(len(peak_x))
    for i in range(len(peak_x)):
        win_width = 100        
        X_win = array[peak_x[i]-win_width:peak_x[i]+win_width, 0]
        Y_win = diffs[peak_x[i]-win_width:peak_x[i]+win_width]
        fwhm = FWHM(X_win, Y_win)
        widths[i] = fwhm
        
        if plot:
            ax.plot([array[peak_x[i], 0],
                     array[peak_x[i], 0] + fwhm],
                    [diffs[peak_x[i]]/2, diffs[peak_x[i]]/2])
    return array[peak_x, 0], widths


# %% Load all time series and find the peaks
folder = 'C:/Users/KlasRydhmer/Documents/Git Repositories/AppStat-Experiments/Ball on Incline/Balls/Original/'
signals, files = load_balls(folder)


plt.close('all')
plot = False
df_peaks = pd.DataFrame(columns=['File', 'Peak_no', 'Time', 'FWHM'])
count = 0
# Loops over all files and finds the peaks
for i in range(len(signals)):
    array = signals[i]
    peak_times, widths = find_peaks(array, plot=plot)

    print(files[i], len(peak_times))

    for j in range(len(peak_times)):
        df_peaks.loc[count, 'File'] = files[i]
        df_peaks.loc[count, 'Peak_no'] = j
        df_peaks.loc[count, 'Time'] = peak_times[j] - peak_times[0]  # Dont miss this!!!
        df_peaks.loc[count, 'FWHM'] = widths[j]
        count += 1
                
    if plot:
        ax = plt.gca()
        ax.set_title(files[i])
    

df_distances = pd.DataFrame()
df_distances['Distance'] = np.array([L1_mean, L2_mean, L3_mean, L4_mean, L5_mean])
df_distances['Sigma'] = np.array([L1_uncert, L2_uncert, L3_uncert, L4_uncert, L5_uncert])

# %% Plot to investigate acceleration for all measurements
plt.close('all')
fig, ax = plt.subplots(1)

for file in files:

    ind = df_peaks['File'] == file
    df_t = df_peaks.loc[ind, :]
    
    if len(df_t) != len(df_distances):
        print('Spooky stuff going on in file', file)
        continue

    x = df_t['Time'].astype(float).values
    y = df_distances['Distance'].values
    
    xe = df_t['FWHM'].astype(float).values
    ye = df_distances['Sigma'].values
    
    if 'small'in file:
        color='blue'
    else:
        color='red'

    ax.errorbar(x, y, xerr=xe, yerr=ye, lw=1, ls='', marker='', c=color)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (mm)')


# %% Weighted mean of times
ind_small = df_peaks['File'].str.contains('small')
ind_big = df_peaks['File'].str.contains('big')

df_peaks.loc[ind_small, ['Peak_no', 'Time', 'FWHM']]

df_peak_summary = pd.DataFrame()
count = 0
for i in range(5):
    ind_peak = df_peaks['Peak_no'] == i
    
    print('Peak: ', i)
    mean, sigma = weighted_error_prop_mean(df_peaks.loc[ind_small & ind_peak, 'Time'],
                                           df_peaks.loc[ind_small & ind_peak, 'FWHM'])
    
    df_peak_summary.loc[count, 'Sphere_size'] = 'small'
    df_peak_summary.loc[count, 'Peak_no'] = i
    df_peak_summary.loc[count, 'Time_mean'] = mean
    df_peak_summary.loc[count, 'Time_sigma'] = sigma
    count += 1

for i in range(5):
    ind_peak = df_peaks['Peak_no'] == i
    
    print('Peak: ', i)
    mean, sigma = weighted_error_prop_mean(df_peaks.loc[ind_big & ind_peak, 'Time'],
                                           df_peaks.loc[ind_big & ind_peak, 'FWHM'])
    
    df_peak_summary.loc[count, 'Sphere_size'] = 'Big'
    df_peak_summary.loc[count, 'Peak_no'] = i
    df_peak_summary.loc[count, 'Time_mean'] = mean
    df_peak_summary.loc[count, 'Time_sigma'] = sigma
    count += 1

# %% Summary plot for X2 fit
plt.close('all')
fig, ax = plt.subplots(1)

"""
TODO!
    ind = df_peaks['File'] == file
    df_t = df_peaks.loc[ind, :]
    
    if len(df_t) != len(df_distances):
        print('Spooky stuff going on in file', file)
        continue

    x = df_t['Time'].astype(float).values
    y = df_distances['Distance'].values
    
    xe = df_t['FWHM'].astype(float).values
    ye = df_distances['Sigma'].values
    
    if 'small'in file:
        color='blue'
    else:
        color='red'

    ax.errorbar(x, y, xerr=xe, yerr=ye, lw=1, ls='', marker='', c=color)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance (mm)')
"""
    
    
# %%

Datlengs=np.zeros(10)
for i in range(10):
    Lenload=np.genfromtxt(os.getcwd()+"/Balls/Original/big_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T  #Skips the header and transposes so first axis is shortest (ie. data along columns)
    Datlengs[i]=len(Lenload[0])                                                                                  #Records the lengths of each data set
minlen=int(np.amin(Datlengs))                                                                                    #Finds the dataset of smallest length

Big_Sphere_O=np.zeros((10,2,minlen))                                                                               #Defines an array with length of the smallest set
for i in range(10):
    Inload=np.genfromtxt(os.getcwd()+"/Balls/Original/big_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T   #Loads a dataset
    Big_Sphere_O[i,:,:]=Inload[:,len(Inload[0])-minlen:]                                                           #Only uses the last minlen entries in each set

#with the same methos, we then load the same data for the small case and then the reverse cases for each.
Datlengs=np.zeros(10)
for i in range(10):
    Lenload=np.genfromtxt(os.getcwd()+"/Balls/Original/small_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T  
    Datlengs[i]=len(Lenload[0])                                                                                 
minlen=int(np.amin(Datlengs))                                                                                    

Small_Sphere_O=np.zeros((10,2,minlen))                                                                               
for i in range(10):
    Inload=np.genfromtxt(os.getcwd()+"/Balls/Original/small_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T   
    Small_Sphere_O[i,:,:]=Inload[:,len(Inload[0])-minlen:]  

#Reverse big
Datlengs=np.zeros(10)
for i in range(10):
    Lenload=np.genfromtxt(os.getcwd()+"/Balls/Reverse/big_rev_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T  
    Datlengs[i]=len(Lenload[0])                                                                                 
minlen=int(np.amin(Datlengs))                                                                                    

Big_Sphere_R=np.zeros((10,2,minlen))                                                                               
for i in range(10):
    Inload=np.genfromtxt(os.getcwd()+"/Balls/Reverse/big_rev_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T   
    Big_Sphere_R[i,:,:]=Inload[:,len(Inload[0])-minlen:] 

#Small rev sphere
Datlengs=np.zeros(10)
for i in range(10):
    Lenload=np.genfromtxt(os.getcwd()+"/Balls/Reverse/small_rev_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T  
    Datlengs[i]=len(Lenload[0])                                                                                 
minlen=int(np.amin(Datlengs))                                                                                    

Small_Sphere_R=np.zeros((10,2,minlen))                                                                               
for i in range(10):
    Inload=np.genfromtxt(os.getcwd()+"/Balls/Reverse/small_rev_sphere_%s.csv"%(i+1),delimiter=',',skip_header=15).T   
    Small_Sphere_R[i,:,:]=Inload[:,len(Inload[0])-minlen:]   

#Next we identify the peaks to find the time values for each pass
def Peaksfromarr(A):                                                                                            #This function takes an array with gate data along axis1 and N experiments along axis0 and returns the right edges of the peaks
    redge_time=np.zeros((A.shape[0],5))
    for i in range(A.shape[0]):
        mask=(A[i,1,:]>=4)                                                                                       #By only looking at values above 4 we sort out the rough area around the peaks
        Peaks=np.vstack((A[i,0,:][mask],A[i,1,:][mask]))                                                         #Then we apply the mask to both the time and voltage data
        dx=(np.roll(Peaks[0,:],-1)-Peaks[0,:])                                                                 #We then look at the difference in measured times between each element and it's neighbour.
        redges=Peaks[0][abs(dx)>=0.05]                                                                               #The right-side edges of each peak will have a larger value and so we find them like this
        if (len(redges)>5):
            redges=redges[:-1]                                                                                   #Removes the last element if there are too many peaks. This is if the ball bounced back into a gate.
        redge_time[i,:]=redges
    return redge_time                                                                                            #It feels bad to not give an uncertainty. Technically I guess this should be related to the extension of the slope for the peak sides, but it seems negligible from the plots and will be dominated by other errors regardless.

#Only thing remaining is to combine the time data with length to fit an a and then combine all into a measure of g