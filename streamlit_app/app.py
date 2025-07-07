import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

st.markdown("""
<style>
body {
    background: linear-gradient(to right, #f7f8fc, #e3f2fd);
}
.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #ffffff, #f1f5f9);
}
h1, h2, h3 {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("streamlit_app/kddcup.data_10_percent_corrected", header=None)



    columns = [f"feature_{i}" for i in range(df.shape[1]-1)] + ["label"]
    df.columns = columns
    return df

@st.cache_resource
def train_model():
    df = load_data()
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    X = df.drop(columns=["label"])
    cat_cols = ['feature_1', 'feature_2', 'feature_3']
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    clf.fit(X_scaled[df['label'] == 0])
    return clf, scaler, X_scaled, df['label'], X.columns.tolist()

model, scaler, X_scaled, y, feature_names = train_model()

st.set_page_config(page_title="🚨 Network Anomaly Detection",  layout="wide")
st.title("🚨 Network Anomaly Detection Dashboard")

with st.sidebar:
    st.title("📊 Data Insights & Settings")
    tab1, tab2 = st.tabs(["Dataset", "Settings"])

    with tab1:
        if st.checkbox("Show Dataset Summary"):
            st.write(load_data().describe())

        if st.checkbox("Show Class Distribution"):
            class_dist = pd.Series(y).value_counts().rename({0: "Normal", 1: "Anomaly"})
            fig = px.pie(values=class_dist, names=class_dist.index, 
                         color_discrete_sequence=px.colors.qualitative.Pastel, 
                         title="Normal vs Anomaly Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Adjust Parameters for Prediction")
        duration = st.number_input("Duration (sec)", min_value=0.0, max_value=100.0, value=0.1, step=0.1)
        src_bytes = st.number_input("Src Bytes", min_value=0, max_value=100000, value=300)
        dst_bytes = st.number_input("Dst Bytes", min_value=0, max_value=100000, value=3000)
        count = st.number_input("Count", min_value=0, max_value=100, value=5)
        srv_count = st.number_input("Srv Count", min_value=0, max_value=100, value=0)

uploaded_file = st.sidebar.file_uploader("📂 Upload your own network data (.csv)", type="csv")
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", user_df.head())


input_features = np.zeros(X_scaled.shape[1])
input_features[0] = duration
input_features[4] = src_bytes
input_features[5] = dst_bytes
input_features[22] = count
input_features[23] = srv_count
scaled_input = scaler.transform([input_features])

with st.spinner('Running anomaly detection...'):
    progress = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.005)
        progress.progress(percent_complete + 1)

score = model.decision_function(scaled_input)[0]
THRESHOLD = 0.01

if score < THRESHOLD:
    prediction = "❌ Anomaly Detected"
    status_color = "#f8d7da"
    status_icon = "🔴"
else:
    prediction = "✅ Normal"
    status_color = "#d4edda"
    status_icon = "🟢"

st.markdown(f"""
<div style="background-color:{status_color};padding:15px;border-radius:10px;">
    <h3>{status_icon} Status: <b>{prediction}</b></h3>
    <p>🔍 <b>Anomaly Score:</b> {score:.4f}</p>
    <p>📏 <b>Threshold:</b> {THRESHOLD}</p>
</div>
""", unsafe_allow_html=True)


st.markdown("##  Model Performance Metrics")


y_scores = -model.decision_function(X_scaled)
fpr, tpr, _ = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)
prec, recall, _ = precision_recall_curve(y, y_scores)
ap_score = average_precision_score(y, y_scores)

fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC={roc_auc:.3f})',
                  labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                  color_discrete_sequence=['#FF6F61'])
fig_roc.add_shape(type='line', line=dict(dash='dash', color='#333333'), x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(fig_roc, use_container_width=True)

fig_pr = px.area(x=recall, y=prec, title=f'Precision-Recall Curve (AP={ap_score:.3f})',
                 labels={'x': 'Recall', 'y': 'Precision'},
                 color_discrete_sequence=['#6B5B95'])
st.plotly_chart(fig_pr, use_container_width=True)




df_scores = pd.DataFrame({'score': y_scores})


threshold = THRESHOLD  
df_scores['label'] = df_scores['score'].apply(lambda x: 'Normal' if x > threshold else 'Anomaly')


fig_score = px.histogram(
    df_scores, 
    x='score', 
    color='label', 
    nbins=50,
    title="Anomaly Score Distribution ",
    color_discrete_map={
        'Normal': 'green',
        'Anomaly': 'red'
    },
    labels={'score': 'Anomaly Score'}
)

st.plotly_chart(fig_score, use_container_width=True)


anomaly_df = pd.DataFrame(X_scaled, columns=feature_names)
anomaly_df['Score'] = y_scores
top_anomalies = anomaly_df.nsmallest(10, 'Score')
st.markdown("### Top 10 Most Anomalous Data Points")
st.dataframe(top_anomalies)


st.markdown("###  Live Anomaly Gauge")
gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=score,
    delta={'reference': THRESHOLD},
    gauge={'axis': {'range': [-0.5, 0.5]},
           'steps': [
               {'range': [-0.5, THRESHOLD], 'color': "green"},
               {'range': [THRESHOLD, 0.5], 'color': "red"}],
           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD}},
    title={'text': "Anomaly Score Gauge"}))
st.plotly_chart(gauge, use_container_width=True)


st.markdown("##  PCA Visualization")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['label'] = y.replace({0: 'Normal', 1: 'Anomaly'})

fig_pca = px.scatter(pca_df, x='PCA1', y='PCA2', color='label',
                     color_discrete_map={'Normal': 'green', 'Anomaly': 'red'},
                     )
st.plotly_chart(fig_pca, use_container_width=True)
