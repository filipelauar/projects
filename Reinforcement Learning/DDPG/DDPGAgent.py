import argparse
import sys
import matplotlib
from torch.functional import norm
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
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
import torch.nn as nn

def normalisation(x,max,min):
    """
        normalisation from -1 to 1 to min to max
    """
    return (max-min)*(x + 1)/2 + min

class DDPGAgent(object):

    def __init__(self, env, opt, discount = 0.99,\
                 learning_rate = 1e-3, freq_opti = 10000,\
                sample_size = 10,capacity = 1000, p = 0.8, nb_update = 1):
        """
        discount : discount reward
        freq_opti : number of optimisation to update the action model
        sample_size : number of iteration in a sample
        entropy: entropy coef to add (if 0 nothing is added)
        """
        self.opt=opt
        self.env=env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.a_min, self.a_max = torch.tensor(self.action_space.low).to(self.device), torch.tensor(self.action_space.high).to(self.device)

        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents = 0
        self.count = 0
        

        # parameters specific to DDPG
        self.nb_update = opt.get("nb_update",nb_update)
        self.p = opt.get("p",p)
        
        self.replay = Memory(opt.get("capacity",capacity), \
            prior=opt.get("prioritized",False),p_upper=1.,epsilon=.01,alpha=1,beta=1)
        self.discount = opt.get("discount",discount)
        self.freq_opti = opt.get("freq_opti",freq_opti)
        self.sample_size = opt.get("sample_size",sample_size)
        self.reward_normalisation = opt.get("reward_normalisation",1)
        self.random_steps = opt.get("random_steps",0)
        self.std = opt.get("std",0)
        self.theta = opt.get("theta",0.15)

        n_in = self.featureExtractor.outSize
        n_out = len(self.a_min)
        
        # self.politique = NN(n_in, n_out, layers=[128,128], finalActivation=torch.nn.Tanh(), final_normalisation = self.a_max, \
        #                    activation=torch.nn.ReLU(),dropout=0.0, normalisation=False)

        self.politique = NN(n_in, n_out, layers=[64,64,32,32], finalActivation=torch.nn.Tanh(), final_normalisation = lambda x: normalisation(x,self.a_max,self.a_min), \
                            activation=torch.nn.ReLU(),dropout=0.0, normalisation=False).to(self.device)
 
        
        # self.critique = NN(n_in + n_out, 1, layers=[128,128], finalActivation=None, \
        #                    activation=torch.nn.ReLU(),dropout=0.0, normalisation=False)

        self.critique = NN(n_in + n_out, 1, layers=[64,64,32,32], finalActivation=None, \
                            activation=torch.nn.ReLU(),dropout=0.0, normalisation=False).to(self.device)


        self.critique_target = copy.deepcopy(self.critique).to(self.device)
        self.politique_target = copy.deepcopy(self.politique).to(self.device)
        
        self.loss = torch.nn.MSELoss() #Pour l'apprentissage
        self.optimizer_politique = torch.optim.Adam(self.politique.parameters(), opt.get("politique_learning_rate",learning_rate))
        self.optimizer_critique = torch.optim.Adam(self.critique.parameters(), opt.get("critique_learning_rate",learning_rate))

        self.noise = Orn_Uhlen(n_actions = len(self.a_min),sigma = self.std, theta=self.theta)


    def act(self, obs):
        self.politique.eval()
        self.critique.eval()
        clip_res=[]


        res = self.politique(torch.tensor(obs,dtype=torch.float).to(self.device)).detach()
        noise = self.noise.sample().to(self.device)

        res = res[0] + normalisation(noise,self.a_max,self.a_min) # normalize noise inside the correct interval
        for i in range(len(self.a_min)):
            clip_res += [torch.clip(res[i],self.a_min[i] ,self.a_max[i])]
        return np.array(torch.tensor(clip_res).detach())

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.critique.state_dict(),outputDir+"_critique")
        torch.save(self.politique.state_dict(),outputDir+"_politique")

    # chargement du modèle.
    def load(self,inputDir):
        self.critique.load_state_dict(torch.load(inputDir+"_critique",map_location=self.device))
        self.politique.load_state_dict(torch.load(inputDir+"_politique",map_location=self.device))


    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        self.politique.train()
        self.critique.train()
        mean_loss = [0, 0]
        for _ in range(self.nb_update):
            
            self.optimizer_critique.zero_grad()

            samples = self.replay.sample(self.sample_size)[2]

            obs = torch.tensor([x[0][0] for x in samples],dtype=torch.float).to(self.device)
            new_obs = torch.tensor([x[3][0] for x in samples],dtype=torch.float).to(self.device)
            actions = torch.tensor([x[1] for x in samples],dtype=torch.float).to(self.device)
            d = torch.tensor([x[4] for x in samples],dtype=torch.float).to(self.device)
            rewards = torch.tensor([x[2] for x in samples],dtype=torch.float).to(self.device)
            Y = rewards + self.discount*(1-d)*self.critique_target(torch.cat((new_obs,self.politique_target(new_obs)), axis = 1)).squeeze(1).detach()
            Y_hat = self.critique(torch.cat((obs,actions),axis=1)).squeeze(1)
            l_crit = self.loss(Y_hat,Y)
            
            l_crit.backward()
            self.optimizer_critique.step()

            mean_loss[0] += l_crit.detach()

            self.optimizer_politique.zero_grad()

            l_pol = -torch.mean(self.critique(torch.cat((obs,self.politique(obs)),axis=1)))

            l_pol.backward()
            self.optimizer_politique.step()
            mean_loss[1] += l_pol.detach()

            for old_param, new_param in zip(self.politique_target.parameters(),self.politique.parameters()):
                old_param.data.copy_(self.p*old_param.data + (1-self.p)* new_param.data)

            for old_param, new_param in zip(self.critique_target.parameters(),self.critique.parameters()):
                old_param.data.copy_(self.p*old_param.data + (1-self.p)* new_param.data) 
  
        return mean_loss[0]/self.nb_update, mean_loss[1]/self.nb_update


    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition

        if not self.test:
            tr = (ob, action, reward/self.reward_normalisation, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            self.replay.store(self.lastTransition)

                    
    
    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if done:
            self.noise.reset()
        if self.test:
            return False
        
        self.nbEvents+=1
        self.count += 1

        return self.count % self.freq_opti == 0 and self.count > self.random_steps
    
