import csv

float_values = [4, 8, 6]
val = [3, 4, 5]
filename = 'float_values.csv'

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Float Values'])  
    writer.writerow(float_values)
    writer.writerow(['Float Values'])  
    writer.writerow(val)


with open(filename, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)  
    float_values_read = [float(value) for value in next(reader)]
    header = next(reader)  
    float_values = [float(value) for value in next(reader)]

print(float_values_read, float_values)  # 
