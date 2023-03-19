
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt



def predict(img):
    image = img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image /= 255
    #### load model here
    model = load_model('MyModel.h5')
    pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
    print("Predicted Number: ", pred.argmax())



predict(cv2.imread('2_2_I_4.jpg'))
