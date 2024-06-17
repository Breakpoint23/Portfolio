from models.AttentionLSTM3.model import AttentionLSTM3

import torch
import numpy as np
from einops import rearrange

MODELPATH="models/AttentionLSTM3/models_saved/model_4.h5"



class model_api():

    def __init__(self,features=5,length=30,hidden=128,modelPath=MODELPATH):

        self.model=AttentionLSTM3(features,length,hidden)

        self.modelPath=modelPath
        self.load_state_dict()



    def load_state_dict(self):
        new_state_dict = {}
        state_dict=torch.load(self.modelPath,map_location=torch.device('cpu'))
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[len('module.'):]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def infer(self,x):

        x=torch.from_numpy(self.normalize(x)).float()
        x=rearrange(x,'b s c -> b c s')

        with torch.inference_mode():
            output=self.model(x)

        return output


    def normalize(self,x):

        x=np.array(x)
        #print("$"*100,"\n",x.shape)
        for i in range(4):
            mi=-2000
            ma=5000
            x[:,:,i]=(x[:,:,i] - mi)/(ma-mi)
            #x=np.swapaxes(x,1,2)

        return x



