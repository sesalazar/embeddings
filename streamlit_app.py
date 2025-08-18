import streamlit as st
import numpy as np
from openai import OpenAI
from scipy.spatial import distance

client = OpenAI(api_key="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")

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
x = st.text_input("Text Field 1:")
y = st.text_input("Text field 2:")
calc = st.button("Calculate")

if calc == True:
    x_embedding = get_embedding(x, model=model)
    y_embedding = get_embedding(y, model=model)
    dist = distance.euclidean(x_embedding,y_embedding)
    st.text("Euclidean distance: " + str(dist))