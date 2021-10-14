import streamlit as st
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from pygam import LogisticGAM
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix, accuracy_score
# import seaborn as sn

#

@st.cache()
def data_prepare():
    data = pd.read_csv('UCI_Credit_Card.csv')
    data = data.iloc[:,1:]
    x = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    columns = data.columns
    num = data.shape[0]
    pred_prob = np.array([])

    return data, x, y, columns, num, pred_prob


def split(x, y, test_size):
    tr_x, te_x, tr_y, te_y = train_test_split(x, y, test_size=test_size)
    return tr_x, te_x, tr_y, te_y

@st.cache()
def model_training(model):
    if model == 'RF':
        model = RandomForestClassifier(n_jobs=-1)
        model.fit(tr_x,tr_y)
        pred_prob = model.predict_proba(te_x)

    elif model == 'DNN':
        model = MLPClassifier(hidden_layer_sizes=(100,100),alpha=0.1)
        model.fit(tr_x,tr_y)
        pred_prob = model.predict_proba(te_x)

    elif model == 'XGB':
        model = XGBClassifier(n_jobs=-1)
        model.fit(tr_x,tr_y)
        pred_prob = model.predict_proba(te_x)

    elif model == 'GAM':
        model = LogisticGAM()
        model.fit(tr_x,tr_y)
        pred_prob = model.predict_proba(te_x)

    elif model == 'GLM':
        model = LogisticRegression()
        model.fit(tr_x, tr_y)
        pred_prob = model.predict_proba(te_x)

    return model, pred_prob

@st.cache()
def corr_map(data):
    map = data.corr().values
    fig2 = px.imshow(map,
                     labels=dict(x="X1", y="X2", color="Productivity"),
                     x=columns,
                     y=columns)
    return fig2


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

#_____________________________navigation_________________________________

page_radio = st.sidebar.radio(
    "Choose Page",
    ('Data Exploration', 'Model Analysis', 'Model Evaluation'))



form = st.sidebar.form(key='my_form')
split_input = form.text_input('test size', '30')

model_selectbox = form.selectbox(
    "choose XAI model",
    ("GLM","GAM","RF", "XGB", "DNN")
)
thres = form.text_input('Threshold value', '0.5')
form.write('Ensemble Tree Model Parameters:', )
tree_num = form.text_input('Tree number:', '')
tree_depth = form.slider('Tree depth:', 1, 10, 1)
form.write('DNN Model Parameters:', )
layer_size = form.text_input('Network Architecture:', '(16,16)')
lr = form.text_input('Learning Rate:', '0.1')

submit_button = form.form_submit_button(label='Train')
print(submit_button)



#_________________________________main page_______________________________________

#_________________________________data exploration_______________________________
data, x, y, columns, num, pred_prob = data_prepare()

if page_radio == 'Data Exploration':
    st.write('Data Exploration')
    left_column, right_column = st.columns(2)
    with left_column:
        fea1_selectbox = st.selectbox(
            "choose X1",
            (tuple(columns))
        )
        fea2_selectbox = st.selectbox(
            "choose X2",
            (tuple(columns))
        )
        show_sample = data.reindex(index=np.random.random_integers(0, data.shape[0], size=5000))
        sample_x = show_sample.loc[:, fea1_selectbox]
        sample_y = show_sample.loc[:, fea2_selectbox]

        fig = go.Figure(data=go.Scatter(x=sample_x, y=sample_y, mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

    with right_column:
        fig2 = corr_map(data)
        st.plotly_chart(fig2, use_container_width=True)



# # _________________________________model analysis_______________________________
elif page_radio == 'Model Analysis':
    st.write('Model Analysis')
    # left_column, right_column = st.columns(2)
    tr_x, te_x, tr_y, te_y = split(x, y, test_size=int(split_input))
    box_selectbox = st.selectbox(
        "choose feature",
        (tuple(columns))
    )

    pos = te_x[te_y == 1]
    neg = te_x[te_y == 0]

    fig5 = go.Figure()
    fig5.add_trace(go.Box(y=pos.loc[:, box_selectbox], name='positive sample',
                         boxpoints="all",marker_color='indianred'))
    fig5.add_trace(go.Box(y=neg.loc[:, box_selectbox], name='negative sample',
                         boxpoints="all",marker_color='lightseagreen'))

    st.plotly_chart(fig5, use_container_width=True)

# _________________________________evaluation_______________________________
elif page_radio == 'Model Evaluation':

    if submit_button:
        tr_x, te_x, tr_y, te_y = split(x,y,test_size=int(split_input))
        model, pred_prob = model_training(model_selectbox)

    st.write('Model Evaluation')
    left_column, right_column = st.columns(2)
    if pred_prob.shape[0] != 0:
        pred = []
        for i in pred_prob[:, -1]:
            if i > float(thres):
                pred.append(1)
            else:
                pred.append(0)
        fpr, tpr, _ = roc_curve(te_y, pred_prob[:, -1])
        fig3 = go.Figure(data=go.Scatter(x=fpr, y=tpr))

        map = confusion_matrix(te_y, pred)
        fig4 = px.imshow(map,
                         labels=dict(x="X1", y="X2", color="Productivity"),
                         x=[0,1],
                         y=[0,1]
                         )

    with left_column:
        if pred_prob.shape[0] != 0:
            st.write('class 0 precision:',precision_score(te_y, pred,pos_label=0))
            st.write('class 0 recall:',recall_score(te_y, pred,pos_label=0))
            st.write('class 1 precision:',precision_score(te_y, pred,pos_label=1))
            st.write('class 1 recall:',recall_score(te_y, pred,pos_label=1))
            st.write('accuracy:',accuracy_score(te_y, pred))
            st.write('auc score:', roc_auc_score(te_y, pred_prob[:,-1]))
            st.plotly_chart(fig4, use_container_width=True)


    with right_column:
        if pred_prob.shape[0] != 0:
            st.plotly_chart(fig3, use_container_width=True)

# # _________________________________model explanation_______________________________
# elif page_radio == 'Model Explanation':
    st.write('Model Explanation')
    # left_column, right_column = st.columns(2)
    # with left_column:

    if pred_prob.shape[0] != 0:
        if model_selectbox == 'RF':
            fi = model.feature_importances_
            fig6 = go.Figure(data=go.Bar(x=columns, y=fi))
            fig6.update_layout(title_text='Feature Importance')

            st.plotly_chart(fig6, use_container_width=True)



