import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from pygam import LogisticGAM

class Model:
    def __init__(self):
        self.data = pd.read_csv('UCI_Credit_Card.csv')
        self.data = self.data.iloc[:,1:]
        self.x = self.data.iloc[:,:-1]
        self.y = self.data.iloc[:, -1]
        self.columns = self.data.columns
        self.num = self.data.shape[0]
        self.pred_prob = np.array([])

    # def model_prepare(self):
    #
    #     self.rf_model = RandomForestClassifier(n_jobs=-1)
    #     self.rf_model.fit(self.tr_x,self.tr_y)

    def split(self,test_size):
        self.tr_x, self.te_x, self.tr_y, self.te_y = train_test_split(self.x, self.y, test_size=test_size)

    def model_training(self,model):
        if model == 'RF':
            self.model = RandomForestClassifier(n_jobs=-1)
            self.model.fit(self.tr_x,self.tr_y)
            self.pred_prob = self.model.predict_proba(self.te_x)

        elif model == 'DNN':
            pass

        elif model == 'XGB':
            pass

        elif model == 'GAM':
            pass

        elif model == 'GLM':
            pass


    def show(self):
        st.title("XAI demo")

#________________________________custom data_______________________________________
        # uploaded_file = st.sidebar.file_uploader("Choose a file")
        # if uploaded_file is not None:
        #     # To read file as bytes:
        #     bytes_data = uploaded_file.getvalue()
        #     st.write(bytes_data)
        #
        #     # To convert to a string based IO:
        #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #     st.write(stringio)
        #
        #     # To read file as string:
        #     string_data = stringio.read()
        #     st.write(string_data)
        #
        #     # Can be used wherever a "file-like" object is accepted:
        #     dataframe = pd.read_csv(uploaded_file)
        #     st.write(dataframe)

#________________________________sidebar________________________________________

        split_input = st.sidebar.text_input('test size', '30')
        self.split(test_size=int(split_input))

        model_selectbox = st.sidebar.selectbox(
            "choose XAI model",
            ("RF", "XGB", "DNN")
        )
        st.sidebar.write(model_selectbox + ' parameter:', )
        if model_selectbox == "RF":
            age = st.sidebar.slider('Tree depth:', 1, 10, 1)
        elif model_selectbox == "DNN":
            title = st.sidebar.text_input('Network depth', '')

        if st.sidebar.button('train'):
            self.model_training(model_selectbox)
            st.sidebar.write(model_selectbox+' model has fitted')

        thres = st.sidebar.text_input('Threshold value', '0.5')

#_________________________________main page_______________________________________

        #_________________________________data exploration_______________________________
        with st.expander("Data exploration"):
            left_column, right_column = st.columns(2)
            with left_column:
                fea1_selectbox = st.selectbox(
                    "choose X1",
                    (tuple(self.columns))
                )
                fea2_selectbox = st.selectbox(
                    "choose X2",
                    (tuple(self.columns))
                )
                show_sample = self.data.reindex(index=np.random.random_integers(0, self.data.shape[0], size=5000))
                sample_x = show_sample.loc[:, fea1_selectbox]
                sample_y = show_sample.loc[:, fea2_selectbox]

                fig = go.Figure(data=go.Scatter(x=sample_x, y=sample_y, mode='markers'))
                st.plotly_chart(fig, use_container_width=True)

            with right_column:
                map = self.data.corr().values
                fig2 = px.imshow(map,
                                 labels=dict(x="X1", y="X2", color="Productivity"),
                                 x=self.columns,
                                 y=self.columns
                                 )
                st.plotly_chart(fig2, use_container_width=True)

        # _________________________________evaluation_______________________________
        with st.expander("Evaluation"):
            left_column, right_column = st.columns(2)
            with left_column:
                if self.pred_prob.shape[0] != 0:
                    pred = []
                    for i in self.pred_prob[:,-1]:
                        if i > float(thres):
                            pred.append(1)
                        else:
                            pred.append(0)

                    st.write('precision score:',precision_score(self.te_y,pred))
                    st.write('recall score:', recall_score(self.te_y, pred))
                    st.write('auc score:', roc_auc_score(self.te_y, self.pred_prob[:,-1]))

            with right_column:
                if self.pred_prob.shape[0] != 0:
                    fpr, tpr, _ = roc_curve(self.te_y, self.pred_prob[:,-1])
                    fig3 = go.Figure(data=go.Scatter(x=fpr, y=tpr))
                    st.plotly_chart(fig3, use_container_width=True)

        # # _________________________________evaluation_______________________________
        # with st.expander("Data exploration"):
        #     left_column, right_column = st.columns(2)
        #     with left_column:
        #         fea1_selectbox = st.selectbox(
        #             "choose X1",
        #             (tuple(self.columns))
        #         )
        #         fea2_selectbox = st.selectbox(
        #             "choose X2",
        #             (tuple(self.columns))
        #         )
        #         show_sample = self.data.reindex(index=np.random.random_integers(0, self.data.shape[0], size=5000))
        #         sample_x = show_sample.loc[:, fea1_selectbox]
        #         sample_y = show_sample.loc[:, fea2_selectbox]
        #
        #         fig = go.Figure(data=go.Scatter(x=sample_x, y=sample_y, mode='markers'))
        #         st.plotly_chart(fig, use_container_width=True)
        #
        #     with right_column:
        #         map = self.data.corr().values
        #         fig2 = px.imshow(map,
        #                          labels=dict(x="X1", y="X2", color="Productivity"),
        #                          x=self.columns,
        #                          y=self.columns
        #                          )
        #         st.plotly_chart(fig2, use_container_width=True)
        #
        # # _________________________________evaluation_______________________________
        # with st.expander("Data exploration"):
        #     left_column, right_column = st.columns(2)
        #     with left_column:
        #         fea1_selectbox = st.selectbox(
        #             "choose X1",
        #             (tuple(self.columns))
        #         )
        #         fea2_selectbox = st.selectbox(
        #             "choose X2",
        #             (tuple(self.columns))
        #         )
        #         show_sample = self.data.reindex(index=np.random.random_integers(0, self.data.shape[0], size=5000))
        #         sample_x = show_sample.loc[:, fea1_selectbox]
        #         sample_y = show_sample.loc[:, fea2_selectbox]
        #
        #         fig = go.Figure(data=go.Scatter(x=sample_x, y=sample_y, mode='markers'))
        #         st.plotly_chart(fig, use_container_width=True)
        #
        #     with right_column:
        #         map = self.data.corr().values
        #         fig2 = px.imshow(map,
        #                          labels=dict(x="X1", y="X2", color="Productivity"),
        #                          x=self.columns,
        #                          y=self.columns
        #                          )
        #         st.plotly_chart(fig2, use_container_width=True)





if __name__ == "__main__":
    mm = Model()
    mm.show()

