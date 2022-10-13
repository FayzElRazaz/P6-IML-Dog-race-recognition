import time
import pickle
import base64
import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_model_and_class_names():
    """ Load model and class_names"""
    model = load_model("./models_efficientnetv2laugmentee2/transfert_efficientnetv2l_modelaugmentee2.h5", compile=True)
    class_names = pickle.load(open("./models_efficientnetv2laugmentee/class_names.save", "rb"))
    return model, class_names

def load_image(img):
    """ transform into array and preprocess image """
    img = img.resize((331,331), Image.ANTIALIAS)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    return img_tensor

def get_prediction(model, img, class_names):
    """ Make prediction using model """
    preds = model.predict(img)
    #preds1 = list(model.predict(img))
    #preds2 = preds1
    #preds2.sort()
    #index = [preds1.index(preds2[-(i+1)]) for i in range(3)]
    pred_label = class_names[np.argmax(preds)]
    #pred_label = [class_names[index[i]] for i in range(3)]
    return pred_label

def main():
    model, class_names = load_model_and_class_names()
    st.title("Détection de la race d'un chien depuis une photo")
    file = st.file_uploader("Veuillez charger une image")
    img_placeholder = st.empty()
    success = st.empty()
    submit_placeholder = st.empty()
    submit=False

    if file is not None:
        with st.spinner("Chargement de l'image.."):  
            model, class_names = load_model_and_class_names()
            img = Image.open(file)
            img_placeholder.image(img, width=331)
        submit = submit_placeholder.button("Lancer la détection de race")

    if submit:
        with st.spinner('Résultat en attente...'):    
            submit_placeholder.empty()
            img_tensor = load_image(img)
            res = get_prediction(model=model, img=img_tensor, class_names=class_names)
            success.success("Race détéctée :  {}".format(res))


if __name__ == "__main__":
    main()



