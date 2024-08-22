# streamlit run CIFAR10ClassifierApp.py


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('cifar10_model.h5')
# classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
#  preprocess >> model  >> preprocess


# Define CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier")
st.write("Upload a colored image to classify it into one of the CIFAR-10 categories.")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None: 
    # Preprocess the image
    img = Image.open(uploaded_file)  # metadata img to tensors    
    img = img.resize((32, 32))  # the same shape like input shape for the model 
    img_array = image.img_to_array(img)   # 
    img_array = np.expand_dims(img_array, axis=0)  # batch 
    # img_array = preprocess_input(img_array) # [1,32,32,3] 
    img_array = img_array.astype('float32') /255.0
    

    # Classify the image
    predictions = model.predict(img_array) # list of proba   [0.1 ,0.0 , 0.5 ,,,, 0.9]
    predicted_class = classes[np.argmax(predictions)] # text 
# np.argmax(predictions) ---> 9 
    # Display the image and the prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Prediction: **{predicted_class}**")
