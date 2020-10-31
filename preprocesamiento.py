import csv
import os

# Generate arrays 
# read the train ids
male = []
female = []
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
            
    print(f'Processed {line_count} lines.')
    

print(f'Male: {len(male)}')
print(f'Female: {len(female)}')


# create directories
directories=['./train','./train/male','./train/female','./test','./test/male','./test/female']
for dirName in directories:
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

for label in range(1, 229):
    try:
        # Create target Directory
        os.mkdir('./train/male/'+str(label))
        os.mkdir('./train/female/'+str(label))
        os.mkdir('./test/male/'+str(label))
        os.mkdir('./test/female/'+str(label))
    except FileExistsError:
        print("Directory already exists")
        break



trainMale=male[:4622]
testMale=male[4622:]

trainFemale=female[:4622]
testFemale=female[4622:]


from shutil import copyfile
# Se copian las imagenes
for i in trainMale:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./train/male/"+str(i[1])+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('TrainMale terminado...')

for i in testMale:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./test/male/"+str(i[1])+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('TestMale terminado...')

for i in trainFemale:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./train/female/"+str(i[1])+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('TrainFemale terminado...')

for i in testFemale:
    src="./input/boneage-training-dataset/boneage-training-dataset/"+str(i[0])+".png"
    dst="./test/female/"+str(i[1])+"/"+str(i[0])+".png"
    copyfile(src, dst)
print('TestFemale terminado...')