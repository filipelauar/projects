
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


from DDPGAgent import *



env, config, outdir, logger = init('./configs/config_random_mountaincar.yaml', "DDPG")

freqTest = config["freqTest"]

freqSave = config["freqSave"]
nbTest = config["nbTest"]
env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = config["nbEpisodes"]
config["maxLengthTrain"] = 200 # 200 for pendulum
config["maxLengthTest"] = 200

agent = DDPGAgent(env, config) 


rsum = 0
mean = 0
verbose = True
itest = 0
reward = 0
done = False
rloss_critique, rloss_politique = 0, 0
mean_obj = 0
mean_B = 0
mean_DKL = 0

for i in range(episode_count):
    checkConfUpdate(outdir, config)

    rsum = 0
    rloss_critique, rloss_politique, mean_obj, mean_B, mean_DKL = 0, 0, 0, 0, 0
    agent.nbEvents = 0
    ob = env.reset()

    # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
    if i % int(config["freqVerbose"]) == 0 and i!=0:
        verbose = True
    else:
        verbose = False

    # C'est le moment de tester l'agent
    if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
        print("Test time! ")
        mean = 0
        agent.test = True

    # On a fini cette session de test
    if i % freqTest == nbTest and i > freqTest:
        print("End of test, mean reward=", mean / nbTest)
        itest += 1
        logger.direct_write("rewardTest", mean / nbTest, itest)
        agent.test = False

    # C'est le moment de sauver le modèle
    if i % freqSave == 0:
        agent.save(outdir + "/save_" + str(i))

    j = 0
    if verbose:
        env.render()

    new_ob = agent.featureExtractor.getFeatures(ob)
    train = False

    while True:
        if verbose:
            env.render()

        ob = new_ob
        action= agent.act(ob) #action
        new_ob, reward, done, _ = env.step(action)
        new_ob = agent.featureExtractor.getFeatures(new_ob)

        j+=1

        # Si on a atteint la longueur max définie dans le fichier de config
        if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
            done = True
            print("forced done!")

        agent.store(ob, action, new_ob, reward, done,j)
        rsum += reward

        
        if agent.timeToLearn(done):
            a, b = agent.learn()
            rloss_politique += a
            rloss_critique += b
            train = True
            # mean_obj += c
            # mean_B += d
            # mean_DKL += e
            

        if done :
            #print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
            if not agent.test:
                logger.direct_write("reward", rsum, i)
                logger.direct_write("itérations", j, i)
                if train:
                    logger.direct_write("Critique/loss", rloss_critique/j, i)
                    logger.direct_write("Politique/loss", rloss_politique/j, i)
                    # logger.direct_write("Politique/Objectif", mean_obj/j, i)
                    # logger.direct_write("Politique/B", mean_B/j, i)
                    # logger.direct_write("Politique/DKL", mean_DKL/j, i)
            agent.nbEvents = 0
            mean += rsum
            rsum = 0

            break
agent.save(outdir + "/save_final")
env.close()
