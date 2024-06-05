import streamlit as st
from streamlit.components.v1 import html

from utils.home import fetch_logs,enroll

st.set_page_config(layout="wide")

ss=st.session_state

st.write("# Meet Mevada")

with open("html/index.html","r") as f:
    htmlComp=f.read()

lCol,rCol=st.columns([2,1])
with lCol:
    html(htmlComp,height=350)

with rCol:
    if st.button("Get logs"):
        fetch_logs(rCol,ss)
    if st.button("enroll"):
        enroll(rCol,ss)
