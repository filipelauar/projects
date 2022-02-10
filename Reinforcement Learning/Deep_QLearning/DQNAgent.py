import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use('TkAgg')

import gym
# import gridworld
import torch
from utils import *
from core import *
import torch
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from memory import *
import copy

class DQNAgent(object):

    def __init__(self, env, opt, capacity = 1000, epsilon = 0.3, gamma = 0.9, \
                 learning_rate = 1e-4,experience_replay = True, freq_opti = 10,\
                sample_size = 10, target_network = True, prioritized = True):
        """
        epsilon : epsilon greedy parameters
        gamma : discount reward
        experience_replay: if True use an experience replay
        target_network: if True, use of a target network
        prioritized: if True, use of a prioritized experience replay
        freq_opti : If target_network is used, frequence of changement of the target network
        sample_size : sample size of replay buffer
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt=opt
        self.env=env

        
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        
        n_in = self.featureExtractor.outSize
        n_out = self.action_space.n
        self.model_action = NN(n_in, n_out, layers=[128], finalActivation=None, \
                            activation=torch.tanh,dropout=0.0).to(self.device)

        self.count = 1
        self.loss = torch.nn.SmoothL1Loss() #Pour l'apprentissage
        self.optimizer_action = torch.optim.Adam(self.model_action.parameters(), opt.get("learning_rate",learning_rate))

         # added parameters
        self.experience_replay = opt.get("experience_replay",experience_replay)
        
        self.epsilon = opt.get("epsilon",epsilon)
        self.gamma = opt.get("gamma",gamma)
        self.sample_size = opt.get("sample_size",sample_size)
        self.target_network = opt.get("target_network",target_network)
        self.freq_opti = opt.get("freq_opti",freq_opti)
        if self.experience_replay:
            self.replay = Memory(opt.get("capacity",capacity), \
                prior=opt.get("prioritized",prioritized),p_upper=1.,epsilon=.01,alpha=1,beta=1)
        if self.target_network:
            self.model_target = copy.deepcopy(self.model_action).to(self.device)

        


    def act(self, obs):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return int(torch.argmax(self.model_action(torch.Tensor(obs).to(self.device))).detach())

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.model_action.state_dict(),outputDir)

    # chargement du modèle.
    def load(self,inputDir):
        self.model_action.load_state_dict(torch.load(inputDir,map_location=self.device))
        

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        self.count += 1
        self.optimizer_action.zero_grad()
        samples = [self.lastTransition] if not self.experience_replay else self.replay.sample(self.sample_size)[2]
        mean_loss = 0
        for ob, a, r, new_ob, d in samples:
            new_ob = torch.Tensor(new_ob).to(self.device)
            r = torch.tensor(r,dtype=torch.float).to(self.device)
            ob = torch.Tensor(ob).to(self.device)
            if d:
                y = r
            else :

                # Calcule de max Q(s', a')
                with torch.no_grad():
                    q_max = (self.model_target(new_ob) if self.target_network else self.model_action(new_ob))
               
                    y = r + self.gamma*torch.max(q_max)
            y_hat = self.model_action(ob)[0][a]
            # print(y_hat.shape,y.shape)
            l = self.loss(y_hat, y)
            l.backward()
            mean_loss += l.detach()
        self.optimizer_action.step()
        
        if self.target_network and self.count % self.freq_opti == 0:
            self.model_target = copy.deepcopy(self.model_action)
        return (mean_loss/len(samples))

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            if self.experience_replay: # store in buffer
                self.replay.store(self.lastTransition)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

