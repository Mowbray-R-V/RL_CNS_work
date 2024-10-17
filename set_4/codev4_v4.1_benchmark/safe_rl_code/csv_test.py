import csv
import os

# Remove the existing file if it exists
if os.path.exists("sim_results.csv"):
    os.remove("sim_results.csv")

# First set of data (for Nikhil)
data1 = [
    {'name': 'Nikhil', 'branch': 'COE', 'year': 2, 'cgpa': 9.0},
]

# Second set of data (for Sanchit)
data2 = [
    {'name': 'Sanchit', 'branch': 'COE', 'year': 2, 'cgpa': 9.1},
]

# Function to prepare rows from the data
def prepare_rows(data):
    return {
        'name': [entry['name'] for entry in data],
        'branch': [entry['branch'] for entry in data],
        'year': [entry['year'] for entry in data],
        'cgpa': [entry['cgpa'] for entry in data]
    }

# Initialize the rows dictionary
rows = prepare_rows(data1)
current_directory = os.getcwd()

# Extract the name of the current directory
current_folder_name = os.path.basename(current_directory)


print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)


print(f'dir:{os.getcwd()}')

# Write the first set of data to the CSV
with open(f'sim_results_{__file__}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    for key, values in rows.items():
        writer.writerow([key] + values)

# print("First file write done")

# # Prepare rows for the second set of data
# new_rows = prepare_rows(data2)

# # Read existing data from the CSV
# with open('sim_results.csv', 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     existing_data = {row[0]: row[1:] for row in reader}

# # Append new data to the existing rows
# for key, values in new_rows.items():
#     existing_data[key].extend(values)

# # Write the updated data back to the CSV
# with open('sim_results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
    
#     for key, values in existing_data.items():
#         writer.writerow([key] + values)

# print("Second file write done")
