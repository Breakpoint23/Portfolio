import requests
import streamlit as st
import pandas as pd
import numpy as np

from utils.model import model_api

INDEXER = pd.api.indexers.FixedForwardWindowIndexer(window_size=30)
FLASK_API_URL = "http://localhost:5000/logs"

MODEL_API=model_api()
def fetch_logs(col1,col2,container,ss):
    response = requests.get(FLASK_API_URL)
    if response.status_code == 200:
        logs = response.json()
        if logs==[]:
            return None
        df=makeDF(logs)

        out=MODEL_API.infer(processData(df))

        print(out.shape)
        dist,passFlag=getDist(ss.enrolledData,out)
        with container:
            with col2:
                st.write(f"#### {passFlag}")
            with col1:
                st.write(f"#### The distance is {np.round(dist,2)}")

            with st.expander("Recived Raw Data"):
                    st.dataframe(df)


    else:
        st.error(f"Failed to fetch logs. Status code: {response.status_code}")


def getDist(x1,x2):
    threshold=3.2
    moreSampleFlag=True if x1.shape[0]>1 else False
    output=0
    passFlag=False

    if x2.shape[0]>1:

        minValue=float("inf")

        for sample in x2:
            dist=(x1-sample).pow(2).sum(axis=1).sqrt()
            if moreSampleFlag:
                dist=dist.mean().item()
            else:
                dist=dist.item()

            if dist<minValue:
                minValue=dist
            output+=dist
        
        if minValue<threshold:
            passFlag=True

        output/=x2.shape[0]

    else:
        dist=(x1-x2).pow(2).sum(axis=1).sqrt()
        if moreSampleFlag:
            dist=dist.mean().item()
        else:
            dist=dist.item()
        
        output+=dist

        if dist<threshold:
            passFlag=True
        
    


    return output,"Your Typing matches" if passFlag else "Your Typing does not match"

def makeDF(logs):
    df={"key":[],"pressTime":[],"liftTime":[],"keyCode":[]}
    for log in logs:
        df["key"].append(log["key"])
        df["keyCode"].append(log["keyCode"])
        df["pressTime"].append(log["pressTime"])
        df["liftTime"].append(log["liftTime"])
    df=pd.DataFrame(df)
    df.sort_values("pressTime",inplace=True)
    return df

def processData(df):

    df.pressTime=df.pressTime.astype(float)
    df.liftTime=df.liftTime.astype(float)
    df.keyCode=df.keyCode.astype(int)

    df.loc[:,'PT1']=df.pressTime.shift(-1)
    df.loc[:,'RT1']=df.liftTime.shift(-1)

    df.loc[:,'HL']=df.liftTime - df.pressTime
    df.loc[:,'IL']=df.PT1 - df.liftTime
    df.loc[:,'PL']=df.PT1 - df.pressTime
    df.loc[:,'RL']=df.RT1 - df.liftTime


    df1=df.iloc[:-1]

    df1=df1[['HL','IL','PL','RL','keyCode']]
    df1.keyCode=df1.keyCode/229.0

    output=[]

    if len(df1)<30:
        data=df1.to_numpy()
        data=np.pad(data,((0,30-len(df1)),(0,0)),mode='constant')
        output.append(data)
        return output

    else:

        for w in df1.rolling(window=INDEXER,step=2):
            if len(w)==30:

                output.append(w.to_numpy())

            else:
                return output
    return output




def enroll(col,ss):
    response = requests.get(FLASK_API_URL)
    if response.status_code == 200:
        logs = response.json()
        if logs==[]:
            return None
        df=makeDF(logs)

        ss.enrolledData=MODEL_API.infer(processData(df))

        with col:
            st.write("Sample Enrolled")

    else:
        st.error(f"Failed to fetch logs. Status code: {response.status_code}")



