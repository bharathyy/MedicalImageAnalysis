import cv2
#for images
from keras.models import load_model
from PIL import Image
import numpy as np
import tensorflow as tf

model=load_model("Brain_Tumor_10epoch.h5")
image=cv2.imread("D:\\Placements\\Preparation\\Project\\Medical Image Analysis\\archive\\pred\\pred45.jpg")
# we have to convert the image into array then resize it
image=Image.fromarray(image)
image=image.resize((64,64))
image=np.array(image)
#print(image)
input_image=np.expand_dims(image,axis=0)
#result=model.predict_classes(input_image)
result = model.predict(input_image)
predicted_class = np.argmax(result, axis=-1)
print(result)
