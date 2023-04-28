import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
#import plotly.graph_objects as go
from PIL import Image
from streamlit_timeline import timeline
import requests
from io import BytesIO
import streamlit.components.v1 as components


speed_url='https://www.youtube.com/watch?v=0KHv4OpGoFA'
with open('events.json') as f:
    events = f.read()
#video=requests.get(speed_url)
#video_bytes=BytesIO(video.content)

st.set_page_config(page_title='Meet Mevada',page_icon='👨‍💻',layout='wide',initial_sidebar_state='auto')
st.sidebar.caption('Wish to connect?')
st.sidebar.write('📧: mevadameet95@gmail.com.com')
pdfFileObj = open('cv2.pdf', 'rb')
st.sidebar.download_button('Want A Traditional Resume? Click here to download',pdfFileObj,file_name='cv2.pdf',mime='pdf')

with st.container():
    left,right=st.columns(2,gap='large')
    with left:
        st.image(Image.open(r'images/Profile.jpg'),width=600)
    with right:
        st.title('Meet Mevada')
        st.markdown('### Data Scientist | Machine Learning Engineer | Deep Learning Engineer')
    
        st.markdown('**A keen learner with publication in the field of deep learning with proven abilities in software development, control systems , Robotics and Automation while being proficient in Python, SQL, C++ and MATLAB/Simulink.**')

st.markdown('---')

timeline(events,height=600)
with st.container():
    st.markdown('## Masters Thesis - Assistance In Natural Setting Using Deep Learning')

     

    left,right=st.columns(2,gap='large')
    with left:
        st.header('Activity Recognition')
        st.image(Image.open(r'images/activity_recognition.jpg'))
        st.caption('Activity Recognition Using Deep Learning')
    with right:
        st.header('Walking Speed Estimation')
        st.video(speed_url)