from mss import mss
import pydirectinput
import cv2 as cv
import numpy as np
import time
from gym import Env
import easyocr
from gym.spaces import Box, Discrete
from PIL import Image
import torch as T

gpu = True if T.cuda.is_available() else False

def easyocr_read(file: str, reader):

    results = reader.readtext(file)
    results = sorted(results, key=lambda x: x[0][0])
    text_results = [x[-2] for x in results]
    easy_output = " ".join(text_results)  
    easy_output = easy_output.strip() 
    return easy_output

if gpu:
    reader = easyocr.Reader(['en'], gpu = True)
else:
    reader = easyocr.Reader(['en'], gpu = False)    

class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape = (1,84, 84), dtype=np.uint8)
        self.action_space = Discrete(2)
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':1400, "height":500}
        self.done_location = {'top':385, 'left':630, 'width':680, "height":100}

    def step(self, action):
        action_map = {
            0:'space',
            1:'no_op'      
        }
        if action!=1: 
            pydirectinput.FAILSAFE = False
            pydirectinput.press(action_map[action])
        done = self.get_done()
        observation  = self.get_observation()
        reward = 1
        info = {}
        if action!=1:
            time.sleep(0.1) 

        return observation, reward, done, info 

    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()


    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]
        gray = cv.cvtColor(raw, cv.COLOR_BGR2GRAY)
        resized = cv.resize(gray, (84, 84))
        channel = np.reshape(resized, (1, 84, 84))
        
        return channel

    def get_done(self):
        done_i = np.array(self.cap.grab(self.done_location))[:,:,:3]


        done_strings = ['G A M E', 'G A H E', '6 A M E', '6AME', '6 A M E']

        done = False
        Image.fromarray(done_i).save('001.png')
        res = easyocr_read('001.png', reader)[:7]
        if res in done_strings:
            done = True

        return done
