import streamlit as st
from streamlit.components.v1 import html

from utils.home import fetch_logs,enroll

st.set_page_config(layout="wide")

ss=st.session_state

st.write("# Typing Pattern Recognition")

st.write(" by Meet Mevada")


with open("html/index.html","r") as f:
    htmlComp=f.read()

with st.container():
    html(htmlComp,height=250)


with st.expander("Instructions"):

    with open("README.md","r") as f:
        instructions=f.read()

    st.markdown(instructions)

resCont=st.container()
with resCont:

    st.write("### Results")
    rCol,lCol=st.columns([1,1])


with st.container():

    rCol1,lCol1=st.columns([1,1]) 

    with lCol1:
        st.button("Identify",on_click=fetch_logs,args=[lCol,rCol,resCont,ss],use_container_width=True)
    
    with rCol1:
        st.button("Enroll",on_click=enroll,args=[rCol,ss],use_container_width=True)
