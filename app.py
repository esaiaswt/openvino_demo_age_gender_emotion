import streamlit as st
import PIL
import cv2
import numpy
import utils
from camera_input_live import camera_input_live
 
st.set_page_config(
    page_title="Openvino Mutlimodal Demo: Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="wide")

st.title("Openvino Mutlimodal Demo: Age/Gender/Emotion :sun_with_face:")
st.markdown('### Using three models: \n 1. face-detection-adas-0001 \n 2. emotions-recognition-retail-0003 \n 3. age-gender-recognition-retail-0013')

input = None 
conf_threshold = float(20)/100 # confidence in float => 20% = 0.2

image = camera_input_live()

uploaded_image = PIL.Image.open(image)
uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)

st.image(visualized_image, channels = "BGR")
