import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt  # showing and rendering figures
import csv

# '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
local_weights_file = 'InceptionV3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

#?---------------------------- ! MODELO 2

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (228, activation='softmax')(x)           

model = Model( pre_trained_model.input, x) 

model.load_weights("TrainModel3")

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 20 using test_datagen generator
testX,testY = next(test_datagen.flow_from_directory(
        "./test",
        target_size=(150, 150),
        batch_size=100,
        #batch_size=4,
        class_mode='sparse'))

predY=model.predict(testX, batch_size = 32, verbose = True)

import plotly.graph_objects as go

X = np.arange(len(testY))
prediccionY=[]
error=0
for i in predY:
  prediccionY.append(np.argmax(i))


for i in range(len(prediccionY)):
  error+=abs(np.argmax(prediccionY[i])-testY[i])

print(f"MAE = {error/len(prediccionY)}")
fig = go.Figure(data=[
    go.Bar(name='Edad real', x=X, y=testY),
    go.Bar(name='Edad predicción', x=X, y=prediccionY)
])


""" 

# Change the bar mode
fig.update_layout(barmode='group')
fig.update_xaxes(type='category')
fig.update_layout(
    title="30 resultados random del modelo 3",
    xaxis_title="# de tests",
    yaxis_title="Edad",
    legend_title="Leyenda",

)
fig.show()
 """


# Flatten the output layer to 1 dimension
x1 = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x1 = layers.Dense(1024, activation='relu')(x1)
# Add a dropout rate of 0.2
x1 = layers.Dropout(0.2)(x1)
# Add a final sigmoid layer for classification
x1 = layers.Dense(1, activation='linear')(x1)


#---------------------------- ! MODELO 1
model1 = Model(pre_trained_model.input, x1)

model1.load_weights("TrainModel1")

test_datagen1 = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 20 using test_datagen generator
testX1, testY1 = next(test_datagen1.flow_from_directory(
    "./test",
    target_size=(150, 150),
    batch_size=50,  # !2611
    # batch_size=4,
    class_mode='sparse'))

predY1 = model1.predict(testX1, batch_size=32, verbose=True)


X1 = np.arange(len(testY1))
prediccionY1 = []
for i in predY1:
    prediccionY1.append(i[0])

error = 0
for i in range(len(testY1)):
    error += abs(predY1[i][0]-testY1[i])
    

""" 
fig1 = go.Figure(data=[
    go.Bar(name='Edad real', x=X1, y=testY1),
    go.Bar(name='Edad predicción', x=X1, y=prediccionY1)
])
# Change the bar mode
fig1.update_layout(barmode='group')
fig1.update_xaxes(type='category')
fig1.update_layout(
    title="30 resultados random del modelo 1",
    xaxis_title="# de tests",
    yaxis_title="Edad",
    legend_title="Leyenda",

)
fig1.show() """


#! Escribimos

mydict=[]
for s in range(len(prediccionY1)):
      mydict.append({
        'Iteracion': s,
        'predMod1': prediccionY1[s],
        'edadMod1': testY1[s],
        'predMod2': prediccionY[s],
        'edadMod2': testY[s],
        },
      )
# field names  
fields = ['Iteracion', 'predMod1', 'edadMod1', 'predMod2','edadMod2']

# name of csv file  
filename = "dataModelos.csv"

# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv dict writer object  
    writer = csv.DictWriter(csvfile, fieldnames = fields)  
        
    # writing headers (field names)  
    writer.writeheader()  
        
    # writing data rows  
    writer.writerows(mydict)  