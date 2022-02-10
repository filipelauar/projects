
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
# import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime

from DQNAgent import *
from randomAgent import * 



env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "DQNAgent")


env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = 100 
config["maxLengthTrain"] = 500 # 200 by default, 500 for lunar
config["maxLengthTest"] = 500
file = "XP/LunarLander-v2/nothing/save_400"

agent = DQNAgent(env, config)
agent.load(file)

mean = []
verbose = True
itest = 0
reward = 0
done = False

agent.test = True

for i in range(episode_count):
    checkConfUpdate(outdir, config)

    rsum = 0
    agent.nbEvents = 0
    ob = env.reset()

    new_ob = agent.featureExtractor.getFeatures(ob)

    j = 0
    while True:
        ob = new_ob
        action= agent.act(ob) #action
        new_ob, reward, done, _ = env.step(action)
        new_ob = agent.featureExtractor.getFeatures(new_ob)

        j+=1

        # Si on a atteint la longueur max définie dans le fichier de config
        if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
            done = True
            print("forced done!")
        rsum += reward
            

        if done:
            #print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
            agent.nbEvents = 0
            mean += [rsum]
            rsum = 0

            break

env.close()
print("Final Result",np.mean(mean),"+/-",np.std(mean))
