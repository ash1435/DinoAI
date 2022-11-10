from env import WebGame
from model import Network   
import sys
import torch as T
import itertools
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import chromedriver_binary

env = WebGame()
model = T.load('best_model/target.pth', map_location='cpu')
try:
        epochs = int(sys.argv[1])
except:
        epochs = 1
episode = 0
driver = webdriver.Chrome(executable_path='./chromedriver.exe')
driver.maximize_window()
try: 
    driver.get('chrome://dino')
except:   
    pass

obs = env.reset()
time.sleep(2)
while episode<epochs:
    action = model.act(obs)

    new_obs, rew, done, _ = env.step(action)
    obs = new_obs
    if done:
        obs = env.reset()
        episode+=1