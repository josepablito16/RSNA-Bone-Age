from PIL import Image
import io
import base64
import numpy as np
import PIL

from io import BytesIO
from keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model


def loadImage(src):
    image_b64 = src.split(",")[1]
    msg = base64.b64decode(image_b64)
    buf = io.BytesIO(msg)
    img = Image.open(buf).convert('RGB')
    #img.thumbnail(size,Image.ANTIALIAS)
    img=img.resize((150,150),PIL.Image.LANCZOS)
    #img.show()
    x = image.img_to_array(img)/255
    #print(x.shape)
    x = np.expand_dims(x, axis=0)
    #print(x.shape)
    images = np.vstack([x])
    return images



def predictModel1(images):
    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = None)

    #pre_trained_model.load_weights(local_weights_file)

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

    model.load_weights("../TrainModel1")

    prediction=model.predict(images,batch_size=10)
    return prediction[0][0]


def predictModel2(images):
    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = None)


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
    x = layers.Dense  (228, activation='softmax')(x)           

    model = Model( pre_trained_model.input, x) 

    model.load_weights("../TrainModel3")
    prediction=model.predict(images,batch_size=10)
    return np.argmax(prediction[0])



#print(predictModel1(loadImage(src)))