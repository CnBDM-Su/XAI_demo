import streamlit as st
import plotly.graph_objs as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pygam import LogisticGAM

class Model:
    def __init__():
        self.data = pd.read_csv('data-additional-full.csv', sep=';')

    def glm(self):

        pass

    def gam(self):

        pass

    def DNN(self):

        pass

    def rf(self):

        pass

    def xgb(self):

        pass




st.title("XAI demo")


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
   # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

   # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

add_selectbox = st.sidebar.selectbox(
    "choose XAI model",
    ("GAMinet", "SHAP", "LIME")
)