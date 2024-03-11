import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf


model=load_model("Brain_Tumor_10epoch.h5")


def predict_image_class(image):
    image = image.resize((64, 64))
    image_array = np.array(image)
    #print(image)
    input_image = np.expand_dims(image_array, axis=0)
    result = model.predict(input_image)
    predicted_class = np.argmax(result, axis=-1)
    print(type(result))

    value_to_compare=1
    all_equal = np.all(result == value_to_compare)

    if all_equal==1:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor Detected")





st.title('Brain Tumor Detection  ')
st.write('Upload an image and click the button to predict.')

# Upload image through file uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg"])

# Display the uploaded image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    
    
    # Make predictions on the uploaded image
    if st.button('Predict'):
        x=predict_image_class(image)
        st.write(x)


        
        
