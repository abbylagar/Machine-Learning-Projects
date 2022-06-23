import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import gradio as gr
from einops import rearrange
import s2cell
from geopy.geocoders import Nominatim


TF_MODEL_URL = 'https://tfhub.dev/google/planet/vision/classifier/planet_v2/1'
IMAGE_SHAPE = (299, 299)
labels=pd.read_csv('planet_v2_labelmap.csv')
classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                 input_shape=IMAGE_SHAPE+(3,)
                                                 )])


def classify_image(image):
    img = image/255.0
    img = rearrange(img, 'h w c  -> 1 h w c')
    prediction = classifier.predict(img)
    s2code = np.argmax(prediction)
    loc=labels['S2CellId'][s2code]
    location=s2cell.token_to_lat_lon(loc)
    geolocator = Nominatim(user_agent="coordinateconverter")
    address = location
    location_add = geolocator.reverse(address)
    return location,location_add



title = 'Photo Geolocation'

description = 'Just upload or drop an image to know where your photo is taken . '

article ='''PlaNet -Photo Geolocation with Convolutional Neural Networks. A gradio demo app for estimation of the address and coordinates of your photo.
<div style='text-align: center;'>PlaNet : <a href='https://tfhub.dev/google/planet/vision/classifier/planet_v2/1' target='_blank'>Model Repo</a> | <a href='https://arxiv.org/pdf/1602.05314v1.pdf' target='_blank'>Paper</a></div>'''


ex1 = 'UnitedKingdom_00019_964966881_426cf82f57_1071_98545448@N00.jpg'
ex2 = 'Tanzania_00019_1292210091_693ea74b7a_1088_15274769@N00.jpg'
ex3 = 'Sydney_00073_1226915900_eea86783cd_1128_65768710@N00.jpg'
ex4 = 'HongKong_00041_504492617_7af38e0004_208_7224156@N03.jpg'
iface = gr.Interface(classify_image, inputs=gr.inputs.Image(shape=(299, 299), image_mode="RGB", type="numpy"),
             outputs=[gr.outputs.Textbox(label='Latitude,Longitude'),gr.outputs.Textbox(label='Address')],examples=[ex1,ex2,ex3,ex4],
            live=False,layout="horizontal", interpretation=None,title=title,
              description=description, article=article)

iface.launch()             