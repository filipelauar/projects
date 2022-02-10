
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")
import gym
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime


from SAC import *



env, config, outdir, logger = init('./configs/config_random_pendulum.yaml', "DDPG")

freqTest = config["freqTest"]

freqSave = config["freqSave"]
nbTest = config["nbTest"]
env.seed(config["seed"])
np.random.seed(config["seed"])
episode_count = config["nbEpisodes"]

config["maxLengthTrain"] = 200 # 200 for pendulum
config["maxLengthTest"] = 200

agent = SACAgent(env, config) 

agent.nbEvents = 0
rsum = 0
mean = 0
verbose = True
itest = 0
reward = 0
done = False
rloss_critique, rloss_politique, rloss_alpha = 0, 0, 0
mean_alpha = 0
mean_B = 0
mean_DKL = 0

for i in range(episode_count):
    checkConfUpdate(outdir, config)

    rsum = 0
    mean_alpha = 0
    
    rloss_critique, rloss_politique, mean_alpha, rloss_alpha = 0, 0, 0, 0
    rloss_crit1, rloss_crit2, rloss_pol, rloss_p, mean_DKL,rloss_qq = 0, 0, 0, 0, 0,0
    log_p = 0
    #agent.nbEvents = 0
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
            #print("forced done!")

        agent.store(ob, action, new_ob, reward, done,j)
        rsum += reward

        
        if agent.timeToLearn(done):
            a, b, c, d,e,f, g = agent.learn()
            rloss_crit1 += a
            rloss_crit2 += b
            rloss_pol += c
            rloss_p += d
            rloss_qq +=e
            rloss_alpha += f
            mean_alpha += g
            train = True

            

        if done :
            #print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
            if not agent.test:
                logger.direct_write("reward", rsum, i)
                logger.direct_write("itérations", j, i)
                if train:
                    logger.direct_write("critique1/Loss", rloss_crit1/j, i)
                    logger.direct_write("critique2/Loss", rloss_crit2/j, i)
                    logger.direct_write("politique/loss", rloss_pol/j, i)
                    logger.direct_write("politique/log_p", rloss_p/j, i)
                    logger.direct_write("politique/cible", rloss_qq/j, i)
                    logger.direct_write("alpha/alpha", mean_alpha/j, i)
                    logger.direct_write("alpha/loss", rloss_alpha/j, i)
            #agent.nbEvents = 0
            mean += rsum
            rsum = 0

            break
agent.save(outdir + "/save_final")
env.close()