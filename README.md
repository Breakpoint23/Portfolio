#### Installation
- Requirements are in requirement.txt file.

1. run the **submit.py** file in api directory which runs flask api to get data from java script and send it to pytorch.

2. Run the `streamlit run home.py` command for starting the web app at **localhost:8501**.

#### Key Press Recorder


This is a simple key press recorder. It records the key press events and sends them to the server.
- The clear text button clears all the recording (except Enrolled sample)

#### Enroll

First, you'll need to enroll (Tell your typing pattern to the model)
1. Type in the text box below
2. Click on the enroll button

#### Identify

This sends your current typing data to the server and compares it with the enrolled data using an AI model called ADELE.

#### Results

It tells you if your typing pattern matches with the enrolled sample.
- It also gives back a distance between two samples (Enrolled and Current) which gives an idea of how different the samples are.

#### Settings

You can change the acceptable ditance between the enrolled sample and new sample using the slider.
