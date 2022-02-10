import matplotlib
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from datetime import datetime
import os
from utils import *


def e_greedy(explo, values_obs, action_space):
    x = np.random.random()
    if x < explo:
        return action_space.sample()
    return np.argmax(values_obs) 
def ucb(explo,values_opt,nb_s_a):
    nb_s = np.sum(nb_s_a)
    return np.argmax(values_opt+explo*np.sqrt((2*np.log(nb_s))/nb_s_a))

def dget(dict,key):
    if dict.get(key) is None:
        dict[key] = 0
    return dict[key]

class QLearning(object):

#tensorboard --logdir XP\gridworld-v0\ --port 6007
    def __init__(self, env, opt):
        self.opt=opt
        self.action_space = env.action_space
        self.env=env
        self.discount=opt.gamma
        self.alpha=opt.learningRate
        self.explo=opt.explo
        self.exploMode=opt.exploMode #0: epsilon greedy, 1: ucb
        self.sarsa=opt.sarsa
        self.modelSamples=opt.nbModelSamples
        self.test=False
        self.qstates = {}  # dictionnaire d'états rencontrés
        self.values = []   # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles
        self.nb_s = {}
        self.dyna = opt.dyna
        self.egi = opt.eligibilite
        if self.egi:
            self.e = {}
            self.lbda = opt.lbda
        if self.dyna:
            self.r = dict()
            self.p = dict()
            self.alphaR = opt.learningRateR
            self.k = opt.k


    def save(self,file):
       pass


    # enregistre cette observation dans la liste des états rencontrés si pas déjà présente
    # retourne l'identifiant associé à cet état
    def storeState(self,obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0) # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
            self.nb_s[ss] = [0] * self.action_space.n
            if self.egi: self.e[ss] = [0] * self.action_space.n
        return ss


    def act(self, obs):
        if self.test:
            return np.argmax(self.values[obs]) 
        
        if self.exploMode == 0:
            return e_greedy(self.explo, self.values[obs], self.action_space)
        elif self.exploMode == 1:
            return ucb(self.explo,self.values[obs],self.nb_s[obs])
 #TODO remplacer par action QLearning

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source=ob
        self.last_action=action
        self.last_dest=new_ob
        self.last_reward=reward
        if it == self.opt.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done=done

        # store r and p for dyna-Q
        s, a, r, s_new = self.last_source, self.last_action, self.last_reward, self.last_dest
        if self.dyna:
            dget(self.r,(s,a,s_new))
            dget(self.p,(s,a,s_new))

    def learn(self, done):
        s, a, r, s_new = self.last_source, self.last_action, self.last_reward, self.last_dest
        if self.exploMode == 1: # update ubc
            self.nb_s[s_new][a] += 1
        if self.egi: # update traces
            
            best_a = np.argmax(self.values[s_new])
            sigma = r + self.discount*self.values[s_new][best_a] - self.values[s][a]
            self.e[s][a] += 1
            for i, s_i in self.e.items():
                for a_j in range(len(s_i)):
                    self.values[i][a_j] += self.alpha*sigma*self.e[i][a_j]
                    self.e[i][a_j] =  self.e[i][a_j]*self.discount*self.lbda + (i==s)
            if done: 
            	for s in self.qstates.values(): 
	            	self.e[s] = [0] * self.action_space.n
        elif self.sarsa:
            a_new = e_greedy(self.explo,self.values[s_new], self.action_space)
            Q_new = self.values[s_new][a_new]
            self.values[s][a] += self.alpha*(r + self.discount*Q_new - self.values[s][a]) 
        else:
            self.values[s][a] += self.alpha * ( r + \
            self.discount*(np.max(self.values[s_new])) - self.values[s][a]) 

        if self.dyna:
            r_value = dget(self.r,(s,a,s_new))
            self.r[(s,a,s_new)] += self.alphaR*(self.last_reward - r_value)
            for q in self.qstates.values():
                p_value = dget(self.p,(s,a,q))
                if q == s_new:
                    self.p[(s,a,q)] += self.alphaR*(1-p_value)
                else:
                    self.p[(s,a,q)] += self.alphaR*(0-p_value)
            for _ in range(self.k):
                a_sampled = self.action_space.sample()
                s_sampled = np.random.choice(list(self.qstates.values()))
                upt = 0
                for q in self.qstates.values():
                    p_value = dget(self.p,(s_sampled,a_sampled,q))
                    r_value = dget(self.r,(s_sampled,a_sampled,q))
                    upt+=p_value*(r_value + self.discount*np.max(self.values[q])) - self.values[s_sampled][a_sampled]
                self.values[s_sampled][a_sampled] += self.alpha*upt
if __name__ == '__main__':
    env,config,outdir,logger=init('./configs/config_qlearning_gridworld.yaml',"QLearning")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]


    agent = QLearning(env, config)

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if (i > 0 and i % int(config["freqVerbose"]) == 0):
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()
        new_ob = agent.storeState(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.storeState(new_ob)

            j+=1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                #print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break



    env.close()
