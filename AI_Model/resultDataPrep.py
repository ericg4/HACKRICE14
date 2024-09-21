import csv
import os
import shutil
import pandas as pd
from collections import defaultdict

def copy_csv_overwrite(source_path, destination_path):
    # Check if the source file exists
    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return

    # Copy the file, overwriting if it already exists
    try:
        shutil.copy2(source_path, destination_path)
        print(f"File successfully copied to '{destination_path}'.")
        if os.path.exists(destination_path):
            print("Existing file was overwritten.")
    except IOError as e:
        print(f"Error copying file: {e}")


def get_csv_column_count(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        first_row = next(csv_reader, None)
        if first_row is not None:
            return len(first_row)
        else:
            return 0


def processCSV(file_path, col_list):
    old_file = os.path.join("CSV_Files", file_path)
    new_file = os.path.join("EditedCSVs", file_path)

    copy_csv_overwrite(old_file, new_file)

    num_columns = get_csv_column_count(new_file)
    # Read the CSV file

    total_list = list(range(num_columns))

    for col in col_list:
        total_list.remove(col)

    with open(new_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    print(f"Limiting file {file_path} to the following columns: \n{data[0]}")

    # Edit the specified cell
    for row in data:
        for index in sorted(total_list, reverse=True):
            if index < len(row):
                del row[index]

    # Write the updated data back to the CSV file
    with open(new_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("Columns removed successfully \n\n")


def compressData(file_path, target_path):

    old_file = os.path.join("EditedCSVs", file_path)
    new_file = os.path.join("EditedCSVs", target_path)


    with open(old_file, 'r') as file:
        reader = csv.reader(file)
        dataCSV = list(reader)

    patient_dict = defaultdict(lambda: [0, 0, 0, 0])

    for row in dataCSV[1:]:
        for idx in range(get_csv_column_count(old_file) - 1):
            if not (row[idx] == " " or row[idx + 1] == " "):
                patient_dict[row[0]][idx] += int(row[idx + 1])
    
    
    os.makedirs(os.path.dirname(new_file), exist_ok=True)

    with open(new_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        data = []

        label_list = dataCSV[0][1:]
        label_list.insert(0, "pnum2")
        data.append(label_list)

        for key, value in patient_dict.items():
            data.append([key, value[0]])
        
        
        # Write the data
        for row in data:
            csv_writer.writerow(row)


def combineDatasets(dataPath1, dataPath2, targetPath):
    print(f"Combining Datasets {dataPath1} and {dataPath2}. ")
    data1 = os.path.join("EditedCSVs", dataPath1)
    data2 = os.path.join("EditedCSVs", dataPath2)

    newCSV = os.path.join("EditedCSVs", targetPath)
    os.makedirs(os.path.dirname(newCSV), exist_ok=True)

    df1 = pd.read_csv(data1)  # The first CSV file
    df2 = pd.read_csv(data2)  # The second CSV file

    # Perform an inner join on the 'ID' column (keeps only rows where 'ID' is common in both dataframes)
    merged_df = pd.merge(df2, df1, on='pnum2', how='inner')
    print(f"Merged Datasets {data1} and {data2} into {newCSV}.\n\n")
    rows, columns = merged_df.shape
    print(f"Size of {newCSV} is H: {rows}, W: {columns}")

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(newCSV, index=False)



# col2_list = [0, 4, 6, 12, 18]
col2_list = [0, 4]
file_path = 'DataSet2.csv'

col1_list = [525, 10, 6, 12, 15, 18, 21, 38, 217, 218, 219, 
            220, 221, 222, 223, 224, 225, 226, 227, 228, 
            229, 230, 231, 232, 233, 234, 41]

# col1_list = list(range(1945))

# exceptionsList = [527,528,529,530,531,532,533,935,936,937,960,978,980,981,982,1449,1450,527,928,950,967,968,1434]
# exceptionsList = [527,528,529,530,531,532,533,935,936,937,960,978,980,981,982,1449,1450]

# for idx in exceptionsList:
#     if idx in col1_list:
#         col1_list.remove(idx)
    

processCSV(file_path, col2_list)
processCSV("DataSet1.csv", col1_list)
compressData("Dataset2.csv", "compressedData.csv")

combineDatasets("DataSet1.csv", "compressedData.csv", "combinedData.csv")
