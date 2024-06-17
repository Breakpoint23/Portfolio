import streamlit as st
from streamlit.components.v1 import html

from utils.home import fetch_logs,enroll,thresholdChange

st.set_page_config(layout="wide")

ss=st.session_state

if "threshold" not in ss:
    ss.threshold=3.2

st.write("# Typing Pattern Recognition")

st.write(" by Meet Mevada")


with open("html/index.html","r") as f:
    htmlComp=f.read()




appTab,settingsTab=st.tabs(["App","Settings"])

with appTab:
    with st.container():
        html(htmlComp)


    with st.expander("Instructions"):

        with open("README.md","r") as f:
            instructions=f.read()

        st.markdown(instructions)

    buttonContainer=st.container()

    resCont=st.container()
    with resCont:

        st.write("### Results")
        rCol,lCol=st.columns([1,1])


    with buttonContainer:

        rCol1,lCol1=st.columns([1,1]) 

        with lCol1:
            st.button("Identify",on_click=fetch_logs,args=[lCol,rCol,resCont,ss],use_container_width=True)
        
        with rCol1:
            st.button("Enroll",on_click=enroll,args=[rCol,ss],use_container_width=True)
with settingsTab:
    st.write("### Settings")
    st.write("Change the threshold value to change the acceptable distance between the typing patterns")
    st.slider("Threshold",min_value=0.1,max_value=10.0,step=0.1,key="threshold",on_change=thresholdChange,args=[ss,"threshold"])