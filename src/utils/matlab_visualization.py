from scipy.io import loadmat
import pandas as pd

# Load the MATLAB file
data = loadmat(r"D:\Thesis_Project\12_lead_ECG_Classification\Code\data\original_data\cpsc_2018\A0001.mat")
print(data)
print(data.keys())
print()

# Access the data using the correct key
ecg_data = data['val']

# Create a DataFrame with the ECG data
print("\nOriginal Dataframe is: ")
df = pd.DataFrame(ecg_data)
print(df)

# Transpose the DataFrame so leads are in columns
df_transposed = df.T

# Rename columns to represent leads
df_transposed.columns = [f'Lead_{i+1}' for i in range(df_transposed.shape[1])]

print("\nTransposed Dataframe with lead number as column names:")
print(df_transposed)
print()