
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

from A2CAgent import *



env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "A2CAgent")


env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = 100 

config["maxLengthTest"] = 500
file = "XP/LunarLander-v2/TDLamdba0.2Entropie/save_final" #
'''
CartPole:
    TD0: 22.75 avant apprentissage
    TD1: 198.27 (3000 itérations)
    TDlambda: 198.49 (5000 itérations)
    Entropy: 199.74
LunarLander:
    TD0: -57.9702562582456 +/- 129.3700621584996
    TD1: 68.04327290630629 +/- 62.440236890347556
    TDLambda: 122.81181557391645 +/- 66.34446334640569 (7000 itérations)
    Entropy: 89.74121494616992 +/- 60.38468413204349 (4000 itérations)
'''
agent = A2CAgent(env, config)
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
        if ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
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
print("Final Result",np.sum(mean)/episode_count,"+/-",np.std(mean))
