import os

# Change this to your CSV file base directory
base_directory = r"C:\Users\GL63\Desktop\2021-05-02-dcvspbks"  
for dir_path, dir_name_list, file_name_list in os.walk(base_directory):
    for file_name in file_name_list:
        # If this is not a CSV file
        if not file_name.endswith('.csv'):
            # Skip it
            continue
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'r',encoding="utf-8") as ifile:
            line_list = ifile.readlines()
        with open(file_path, 'w',encoding="utf-8") as ofile:
            ofile.writelines(line_list[2:])