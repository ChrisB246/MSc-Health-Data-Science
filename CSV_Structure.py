"""
Christopher Brook
10603660
MSc Health Data Science

"""
# imports for this project
import pandas as pd
import glob
from tqdm import tqdm

#importing the dataset
xray_ds = pd.read_csv('C:/MSCDISS/Data_Entry_2017.csv')

# initiating the lists
temp_location = []
temp_path = []

# looping through and collecting the file location of the images.
for img_name in tqdm(xray_ds['Image Index']):
    location = glob.glob('C:/MSCDISS/images_0**/images/' + str(img_name), recursive=True)
    path = str(location)
    path = ''.join(path.split())[:-18]
    path = ''.join(path.split())[2:]
    temp_location.append(location)
    temp_path.append(path)

# setting the list to a dataframe
df_location = pd.DataFrame(data=temp_location)
df_path = pd.DataFrame(data=temp_path)

# renaming the file location
df_location.columns = ['location']
df_path.columns = ['img_path']

# replacing the // with a single / so its in the correct structure
df_location['location'] = df_location['location'].astype(str).str.replace("\\", "/")
df_path['img_path'] = df_path['img_path'].astype(str).str.replace("\\", "/")
df_path['img_path'] = df_path['img_path'].astype(str).str.replace("//", "/")

# concat the dataframes into one column
final_df = pd.concat([df_path, df_location, xray_ds['Finding Labels'], xray_ds['Patient Gender'], xray_ds['Patient Age']], axis=1, ignore_index=False)

# writing final df to a csv.
print("writing")
final_df.to_csv('FULL_Data_Entry_2017_updated.csv', index=False)
print("done")
