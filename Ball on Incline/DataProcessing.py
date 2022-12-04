import pandas as pd
import numpy as np
import os



file_path = str(os.getcwd() + r"/BallInclineData.csv")


df = pd.read_csv(file_path)

r_bball_mean, r_sball_mean, inverse_sum_sigs_bball, inverse_sum_sigs_sball, L1, L2, L3, L4, L5 = 0, 0, 0, 0, 0, 0, 0, 0, 0

# This section all calculates errors using a weighted mean: ------------------------------------------------------------------------------------------

for i in range(len(df['r_bball_sig (mm)'].tolist())):
    inverse_sum_sigs_bball += (1/(df['r_bball_sig (mm)'].dropna().tolist()[i]**2))
    inverse_sum_sigs_sball += (1/(df['r_sball_sig (mm)'].dropna().tolist()[i]**2))


for i in range(len(df['r_bball (mm)'].tolist())):
    r_bball_mean += (df['r_bball (mm)'].dropna().tolist()[i]/(df['r_bball_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_bball
    r_sball_mean += (df['r_sball (mm)'].dropna().tolist()[i]/(df['r_sball_sig (mm)'].dropna().tolist()[i]**2)) / inverse_sum_sigs_sball

uncert_on_r_bball_mean = np.sqrt(1/inverse_sum_sigs_bball)
uncert_on_r_sball_mean = np.sqrt(1/inverse_sum_sigs_sball)

print("The radius of the big sphere is: ", f"{r_bball_mean:.2f}", " +- " , f"{uncert_on_r_bball_mean:.2f}", " with propagated errors")
print("The radius of the small sphere is: ", f"{r_sball_mean:.2f}", " +- " , f"{uncert_on_r_sball_mean:.2f}", " with propagated errors")
