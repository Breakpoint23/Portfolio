import streamlit as st
from streamlit.components.v1 import html
import requests

st.set_page_config(layout="wide")

st.write("# Meet Mevada")

with open("html/index.html","r") as f:
    htmlComp=f.read()

with st.container():
    html(htmlComp,height=1000)
