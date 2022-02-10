
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import numpy as np

from DDPGAgent import *





env, config, outdir, logger = init('./configs/config_random_mountaincar.yaml', "DDPG")


env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = 100 


config["maxLengthTest"] = 200


nums = [str(i) for i in reversed(range(0,1100,100))] # + ["final"]
best = ("None",[-np.inf])
for num in nums:
    file = "XP/MountainCarContinuous-v0/basic/save_"+num

    agent = DDPGAgent(env, config) # PPOAgent_2(env, config)
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
        # env.render()
        while True:
            # env.render()
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
    print("Final Result",np.mean(mean),"+/-",np.std(mean))
    if np.mean(best[1]) < np.mean(mean):
        best = (num,mean)
print("Final Best Result",best[0],np.mean(best[1]),"+/-",np.std(best[1]))