import os
import glob
import pandas as pd
#csv_dir = os.getcwd()
os.chdir(r"C:\Users\GL63\Desktop\2021-05-02-dcvspbks")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(all_filenames)

all_df = []
for f in all_filenames:
    try:
        df = pd.read_csv(f)
        df['file'] = f.split('/')[-1]
        all_df.append(df)
    except pd.errors.EmptyDataError:
        print(f)

merged_df = pd.concat(all_df, ignore_index=True, sort=True)
merged_df.to_csv( "combined_csv1.csv", index=False, encoding='utf-8')