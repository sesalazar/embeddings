import streamlit as st
import numpy as np
from openai import OpenAI
from scipy.spatial import distance

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def get_client():
    return OpenAI()

def get_embedding(text, model):
    edited_text = text.replace("\n"," ")
    return client.embeddings.create(input=[edited_text], model=model).data[0].embedding

st.set_page_config(page_title="Embedding Distance")

#Sidebare: settings to select model
st.sidebar.header("Settings")
model = st.sidebar.selectbox(
    "Embedding model",
    options=[
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
    index = 0,
)

st.write("## Distance Calculator")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### List 1")
    x1 = st.text_input("Text Field 1:")
    x2 = st.text_input("Text Field 2:")
    x3 = st.text_input("Text Field 3:")
    x4 = st.text_input("Text Field 4:")
    x5 = st.text_input("Text Field 5:")

with col2:
    st.markdown("##### List 2")
    y1 = st.text_input("Text Field 6:")
    y2 = st.text_input("Text Field 7:")
    y3 = st.text_input("Text Field 8:")
    y4 = st.text_input("Text Field 9:")
    y5 = st.text_input("Text Field 10:")

calc = st.button("Calculate")

z1 = ""
z2 = ""
z3 = ""
z4 = ""
z5 = ""

if calc == True:
    if x1 != "":
        x1_embedding = get_embedding(x1, model=model)
        y1_embedding = get_embedding(y1, model=model)
        dist = distance.euclidean(x1_embedding,y1_embedding)
        z1 = (str(dist))
    if x2 != "":
        x2_embedding = get_embedding(x2, model=model)
        y2_embedding = get_embedding(y2, model=model)
        dist2 = distance.euclidean(x2_embedding,y2_embedding)
        z2 = (str(dist2))
    if x3 != "":
        x3_embedding = get_embedding(x3, model=model)
        y3_embedding = get_embedding(y3, model=model)
        dist3 = distance.euclidean(x3_embedding,y3_embedding)
        z3 = (str(dist3))
    if x4 != "":
        x4_embedding = get_embedding(x4, model=model)
        y4_embedding = get_embedding(y4, model=model)
        dist4 = distance.euclidean(x4_embedding,y4_embedding)
        z4 = (str(dist4))
    if x5 != "":
        x5_embedding = get_embedding(x5, model=model)
        y5_embedding = get_embedding(y5, model=model)
        dist5 = distance.euclidean(x5_embedding,y5_embedding)
        z5 = (str(dist5))
        

with col3:
    st.markdown("##### Euclidean Distances")
    st.text("")
    st.text("")
    st.text("")
    st.text(z1)
    st.text("")
    st.text("")
    st.text(z2)
    st.text("")
    st.text("")
    st.text("")
    st.text(z3)
    st.text("")
    st.text("")
    st.text("")
    st.text(z4)
    st.text("")
    st.text("")
    st.text(z5)
