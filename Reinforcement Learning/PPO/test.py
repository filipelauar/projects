
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
# import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime

from PPOAgentKLAdaptatif import *
from PPOAgentClipped import *





env, config, outdir, logger = init('./configs/config_random_lunar.yaml', "All")


env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = 100 

config["maxLengthTest"] = 200
all_test = [str(x) for x in range(0,3000,100)] + ["final"]
all_test_res = []
all_experiment = []
max = ([-10000], "")

names = ["KLAdaptatif","basic","Clipped"]
agents = [PPOAgent_1,PPOAgent_1,PPOAgent_2]
for k, name_file in enumerate(names):
    all_test_res = []
    for num in all_test:
        file = "XP/LunarLander-v2/" + name_file + "/save_" + num

        agent = agents[k](env, config) # PPOAgent_2(env, config)
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
        print("Final Result",num,np.mean(mean),"+/-",np.std(mean))
        all_test_res += [mean]
        if np.mean(max[0]) < np.mean(mean):
            max = (mean,num)

    print("Final Result",max[1],np.mean(max[0]),"+/-",np.std(max[0]))
    all_experiment += [all_test_res]

courbes = []

for i in range(len(all_experiment)):
    courbes += plt.plot(range(0,3100,100),np.mean(all_experiment[i],axis=1))
print(courbes)
plt.legend(courbes,names)
plt.ylabel("Rewards Moyen sur 100 épisodes")
plt.xlabel("Nombre d'épisodes en train")
plt.show()
