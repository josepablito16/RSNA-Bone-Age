import csv
import os
import numpy as np

# Generate arrays 
# read the train ids
male = []
female = []
edades=[]
with open('input/boneage-training-dataset.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            row[1]= int(row[1])
            if(row[2] == "True"):
                male.append(row[:2])
            else:
                female.append(row[:2])
            line_count += 1
        edades.append(row[1])
    print(f'Processed {line_count} lines.')

edades=np.array(edades)
print(edades)
for label in range(1, 229):
    print(f"Cantidad de label {label} = {np.sum(edades==str(label))}")
    
