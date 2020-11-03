import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt # showing and rendering figures


local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='linear')(x)           

model = Model( pre_trained_model.input, x) 

model.load_weights("TrainModel1")

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 20 using test_datagen generator
testX,testY = next(test_datagen.flow_from_directory(
        "./test",
        target_size=(150, 150),
        batch_size=2611,
        #batch_size=4,
        class_mode='sparse'))

predY=model.predict(testX, batch_size = 32, verbose = True)

import plotly.graph_objects as go

X = np.arange(len(testY))
prediccionY=[]
for i in predY:
  prediccionY.append(i[0])

error=0
for i in range(len(testY)):
  error+=abs(predY[i][0]-testY[i])

print(f"MAE = {error/len(testY)}")


fig = go.Figure(data=[
    go.Bar(name='Edad real', x=X, y=testY),
    go.Bar(name='Edad predicción', x=X, y=prediccionY)
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_xaxes(type='category')
fig.update_layout(
    title="30 resultados random del modelo 1.2",
    xaxis_title="# de tests",
    yaxis_title="Edad",
    legend_title="Leyenda",

)
fig.show()
'''

data = [[30, 25, 50, 20],
[40, 23, 51, 17],
[35, 22, 45, 19]]
X = np.arange(len(testY))
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, testY, color = 'b', width = 0.25, label = 'real')
ax.bar(X + 0.25, prediccionY, color = 'g', width = 0.25,label = 'predictions')
ax.legend()
ax.set_xlabel('Number of test')
ax.set_ylabel('Age (Months)')
plt.show()
'''







    