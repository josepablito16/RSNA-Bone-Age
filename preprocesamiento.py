import csv
import os

# Generate arrays 
# read the train ids
datos = []
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
            datos.append(row[:2])
            edades.append(row[1])
            line_count += 1
            
    print(f'Processed {line_count} lines.')

print(f"Largo de data {len(datos)}")



# create directories
directories=['./train','./test']
for dirName in directories:
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")



train=datos[:10000]
test=datos[10000:]
trainEdades=edades[:10000]
testEdades=edades[10000:]


for label in list(set(trainEdades)):
    try:
        # Create target Directory
        os.mkdir('./train/'+str(int(label)/228))
        #os.mkdir('./train/'+str(int(label)))
    except FileExistsError:
        print("Directory already exists")
        break

for label in list(set(testEdades)):
    try:
        # Create target Directory
        os.mkdir('./test/'+str(int(label)/228))
        #os.mkdir('./test/'+str(int(label)))
    except FileExistsError:
        print("Directory already exists")
        break




from shutil import copyfile
# Se copian las imagenes
for i in train:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./train/"+str(int(i[1])/228)+"/"+str(i[0])+".png"
    #dst="./train/"+str(int(i[1]))+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('Train terminado...')

for i in test:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./test/"+str(int(i[1])/228)+"/"+str(i[0])+".png"
    #dst="./test/"+str(int(i[1]))+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('TestMale terminado...')
