import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from glob import glob
from PIL import Image
#from tf.keras.metrics.Accuracy import 
#from tf.keras.mertics import percision

DATASET_FOLDER = '/content/drive/MyDrive/QR_d_best_TRAIN/'
dataset_path_test = '/content/drive/MyDrive/QR_d_best_TESTT/'

batch_size = 64
num_classes = 5
epochs = 100

def calculate_label(img_basename):
    #this number (39) It changes depending on the dataset folder address character
    tmp = img_basename[39:len(img_basename)-4].split("_")
    row = tmp[2]
    rod_id = int(tmp[3])
    if row in {"A", "B", "E", "F", "I", "J"}:
        label = 4-(rod_id % 4)
    else:
        label = (rod_id % 4)+1    
    return label



# input data

x_train = np.asarray([np.asarray(Image.open(file).resize((28, 28))) for file in glob(DATASET_FOLDER+'*.jpg')])
print ("Input: " + str(x_train.shape))

y_train = np.asarray([np.asarray(calculate_label(file)) for file in glob(DATASET_FOLDER+'*.jpg')])
print("Input: " + str(y_train.shape))
  
x_test = np.asarray([np.asarray(Image.open(file).resize((28, 28))) for file in glob(dataset_path_test+'*.jpg')])
y_test = np.asarray([np.asarray(calculate_label(file)) for file in glob(dataset_path_test+'*.jpg')])

print(x_train.shape, y_train.shape)



num_classes = 5
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

####Converts a class vector (integers) to binary class matrix.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Create model
# Building CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

##### compile model
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
# if you want percision use metrics=[tf.keras.metrics.Precision()])  




hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=2,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('MyModel.h5')
print("Saving the model as mnist.h5")



score = model.evaluate(x_test, y_test, verbose=0)
#print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
