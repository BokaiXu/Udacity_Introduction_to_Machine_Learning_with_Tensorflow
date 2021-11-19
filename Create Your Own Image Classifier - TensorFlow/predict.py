"""
This is a image calssisfier.
image_convert(url) is used to convert the image to a numpy array.
load_model() is used to load the model I trained.
predict(url, model, top_k) is used to load the numpy array from image_convert(url) and predict the type of the flower.
python predict.py /path/image.jpg model_20.h5 In this model only model_20.h5 could be used.
--top_k K, K is num of predictions
--category_names label_map.json, print out the name of flowers according to the label
"""

# import packages
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging
logger=tf.get_logger()
logger.setLevel(logging.ERROR)
import json
from PIL import Image
import argparse

def image_convert(url):
    image = Image.open(url)
    image_numpy = np.asarray(image)
    image_tf32=tf.cast(image_numpy,tf.float32)
    image_resize=tf.image.resize(image_tf32,(224,224))
    image_resize/=255
    image_resize_numpy= image_resize.numpy()
    return  image_resize_numpy

def load_model():
    return tf.keras.models.load_model('./model_20.h5', custom_objects={'KerasLayer':hub.KerasLayer})

def predict(url, model, top_k):

    image_resize_numpy=image_convert(url)
    image_final=np.expand_dims(image_resize_numpy, axis=0)
    ps=model.predict(image_final)
    probs=tf.nn.top_k(ps, top_k)[0].numpy()
    classes=tf.nn.top_k(ps, top_k)[1].numpy()

    print('The label of the flower in the json map is: ', classes[0])
    print('The probability is: ',list(map('{:.2f}%'.format,probs[0][:top_k]*100)))

    if args.category_names!=None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        name=[]
        for num in classes[0]:
            name.append(class_names[str(num+1)])
        print ('The name of the flower is: ', name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the type of a flower.')
    parser.add_argument('url', type=str, help='path of the image')
    parser.add_argument('model', help='model used to predict the type')
    parser.add_argument('--top_k', type=int, help='top K types of prediction',default=5)
    parser.add_argument('--category_names', help='json file')
    args = parser.parse_args()

    model=load_model()
    url=args.url
    top_k=args.top_k
    predict(url,model,top_k)
