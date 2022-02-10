
import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
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
import torch.nn.functional as F
from itertools import chain

import pdb

class SACAgent(object):

    def __init__(self, env, opt, discount = 0.99,\
                 learning_rate = 1e-4, freq_opti = 10000,\
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


        # parameters specific to SAC
        
        self.nb_update = opt.get("nb_update",nb_update)
        self.p = opt.get("p",p)
        self.replay = Memory(opt.get("capacity",capacity), \
            prior=opt.get("prioritized",False),p_upper=1.,epsilon=.01,alpha=1,beta=1)
        self.discount = opt.get("gamma",discount)
        self.alpha = opt.get("alpha",0.2)
        self.freq_opti = opt.get("freq_opti",freq_opti)
        self.sample_size = opt.get("sample_size",sample_size)
        self.adaptatif = opt.get("adaptatif",False)
        actor_learning_rate = opt.get("politique_learning_rate",learning_rate)
        criti_learning_rate = opt.get("critique_learning_rate",learning_rate)
        lr_alpha = opt.get("learning_rate_alpha",1)
        self.random_sampling = opt.get("random_sampling",0)
        self.reward_normalisation = opt.get("reward_normalisation",1)
        self.separate_mean_std = opt.get("separate",True)
        self.clipped_std = opt.get("clipped_std",False)
        if self.adaptatif:
            self.seuil = self.alpha
            self.alpha = torch.tensor(self.alpha,requires_grad = True).to(self.device)
            

        n_in = self.featureExtractor.outSize
        n_out = len(self.a_min)
        
        self.nb_act = n_out
    
        
        if self.separate_mean_std:
            self.politique_mean = NN(n_in, n_out, layers=[30,30], \
                                activation=torch.nn.LeakyReLU(),dropout=0.0,normalisation=True).to(self.device)

            self.politique_std = NN(n_in, n_out, layers=[30,30], \
                                activation=torch.nn.LeakyReLU(),dropout=0.0,normalisation=True,init_w = [-3e-3, 3e-3]).to(self.device)
        else:
            self.politique = NN(n_in, n_out*2, layers=[30,30], \
                                activation=torch.nn.LeakyReLU(),dropout=0.0,normalisation=True).to(self.device)
        
        self.critique_1 = NN(n_in + n_out, 1, layers=[30,30], \
                            activation=torch.nn.LeakyReLU(),dropout=0.0,normalisation=True).to(self.device)

        self.critique_2 = NN(n_in + n_out, 1, layers=[30,30], \
                            activation=torch.nn.LeakyReLU(),dropout=0.0,normalisation=True).to(self.device)

        # self.startEvents = opt.get('random_steps',10000)
        self.critique_target_1 = copy.deepcopy(self.critique_1).to(self.device)
        self.critique_target_2 = copy.deepcopy(self.critique_2).to(self.device)
        if self.separate_mean_std:
            self.optimizer_politique = torch.optim.Adam(chain(self.politique_std.parameters(),self.politique_mean.parameters()), actor_learning_rate)
        else:
            self.optimizer_politique = torch.optim.Adam(self.politique.parameters(), actor_learning_rate)
            
        self.optimizer_critique_1 = torch.optim.Adam(self.critique_1.parameters(), criti_learning_rate)
        self.optimizer_critique_2 = torch.optim.Adam(self.critique_2.parameters(), criti_learning_rate)
        if self.adaptatif:
            self.optimizer_alpha = torch.optim.Adam([self.alpha], lr=lr_alpha)
        """
        assert self.adaptatif and self.seuil == 0.2 and self.nb_update == 10 and\
             self.freq_opti == 1000 and self.p == 0.9 and self.reward_normalisation == 100 and\
                  actor_learning_rate == 0.001 and criti_learning_rate == 0.01 and lr_alpha == 0.001 and self.sample_size == 1000 \
                  and opt.get("capacity",capacity) == 1000000 
        """

    def act(self, obs):
        # if self.nbEvents<=self.startEvents:
        #        d=np.clip(torch.randn((self.action_space.shape)).numpy(),self.action_space.low,self.action_space.high)
      
        self.critique_1.eval()
        self.critique_2.eval()
        if self.separate_mean_std:
            self.politique_mean.eval()
            self.politique_std.eval()
        else:
            self.politique.eval()
        
        obs = torch.tensor(obs).float().view(1,-1).to(self.device)
        with torch.no_grad():
            a, _ = self.getAction(obs)
    
        return np.clip(np.array(a.detach().cpu()).reshape(-1),self.action_space.low,self.action_space.high)

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.critique_1.state_dict(),outputDir+"_critique_1")
        torch.save(self.critique_2.state_dict(),outputDir+"_critique_2")
        if self.separate_mean_std:
            torch.save(self.politique_std.state_dict(),outputDir+"_politique_std")
            torch.save(self.politique_mean.state_dict(),outputDir+"_politique_mean")
        else:
            torch.save(self.politique.state_dict(),outputDir+"_politique")

    # chargement du modèle
    def load(self,inputDir):
        self.critique_1.load_state_dict(torch.load(inputDir+"_critique_1",map_location=self.device))
        self.critique_2.load_state_dict(torch.load(inputDir+"_critique_2",map_location=self.device))
        if self.separate_mean_std:
            self.politique_std.load_state_dict(torch.load(inputDir+"_politique_std",map_location=self.device))
            self.politique_mean.load_state_dict(torch.load(inputDir+"_politique_mean",map_location=self.device))
        else:
            self.politique.load_state_dict(torch.load(inputDir+"_politique",map_location=self.device))
        

            
    def getAction(self,s):
        if self.separate_mean_std:
            # print(self.politique_mean.layers[0].weight)
            mean = self.politique_mean(s) 
            
            std = torch.exp(self.politique_std(s))
        else:
            m_s = self.politique(s)
            mean = m_s[:,:self.nb_act]
            std = torch.exp(m_s[:,self.nb_act:])
        if self.clipped_std:
            std = torch.clamp(std,0,0.8)
        # mean = torch.clamp(std,-1,1)
        # if mean.shape[0] == 1:
        #    print(mean,std)
        normal = torch.distributions.Normal(mean,std)
        u = normal.rsample()
        log_p = normal.log_prob(u).to(self.device)
        u = torch.tanh(u)
        log_p_x = log_p.sum(-1)-torch.log(1-u*u).sum(-1)
        return u,log_p_x

    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        # print('learn')
        self.critique_1.train()
        self.critique_2.train()
        if self.separate_mean_std:
            self.politique_mean.train()
            self.politique_std.train()
        else:
            self.politique.train()

        mean_loss = [0, 0, 0,0,0, 0]
        

        for _ in range(self.nb_update):


            samples = self.replay.sample(self.sample_size)[2]
            obs = torch.tensor(np.array([x[0][0] for x in samples]),dtype=torch.float).to(self.device)
            new_obs = torch.tensor(np.array([x[3][0] for x in samples]),dtype=torch.float).to(self.device)
            rewards = torch.tensor(np.array([x[2] for x in samples]),dtype=torch.float).to(self.device)
            done = torch.tensor(np.array([x[-1] for x in samples]),dtype=torch.float).to(self.device)
            actions = torch.tensor(np.array([x[1] for x in samples]),dtype=torch.float).view(len(samples),-1).to(self.device)

            # update critic
            
            next_a, log_p_next_a = self.getAction(new_obs)

            
            # next actions score according to politic for next actions
            critique_1 = self.critique_target_1(torch.cat((new_obs, next_a),axis = 1)).view(-1)
            critique_2 = self.critique_target_2(torch.cat((new_obs, next_a),axis = 1)).view(-1)

            V = torch.min(critique_1,critique_2) - self.alpha*log_p_next_a # calculation of V with 

            Y = (rewards + self.discount*(1-done)*V).detach().to(self.device) # what we search
            # excepted values for the two critics
            Y_hat_1 = self.critique_1(torch.cat((obs,actions),axis=1)).view(-1)
            Y_hat_2 = self.critique_2(torch.cat((obs,actions),axis=1)).view(-1)

            l_crit_1 = F.mse_loss(Y_hat_1,Y) 
            l_crit_2 = F.mse_loss(Y_hat_2,Y) 

            # update networks
            self.optimizer_critique_1.zero_grad()
            l_crit_1.backward()
            self.optimizer_critique_1.step()
            
            self.optimizer_critique_2.zero_grad()
            l_crit_2.backward()
            self.optimizer_critique_2.step()

            # update actor

            a, log_p_a = self.getAction(obs) # what the actor think will be action for current observations

            with torch.no_grad():
                critique_1 = self.critique_1(torch.cat((obs, a),axis = 1)).view(-1)
                critique_2 = self.critique_2(torch.cat((obs, a),axis = 1)).view(-1)
                
                q = torch.min(critique_1 ,critique_2)
            
            # print(critique_1.shape,critique_2.shape,q.shape,log_p_a.shape,(q - self.alpha*log_p_a).shape)
            
            l_pol = - torch.mean(q - self.alpha*log_p_a) # our actions should maximise q 

            # update network
            self.optimizer_politique.zero_grad()
            l_pol.backward()
            # print("-------------------------------")
            # print(l_pol)
            # print(self.politique_mean.layers[0].weight)
            self.optimizer_politique.step()
            # print(self.politique_mean.layers[0].weight)
            # print("-------------------------------")

            
            mean_loss[0] += l_crit_1.detach() 
            mean_loss[1] += l_crit_2.detach()
            mean_loss[2] += l_pol.detach()
            mean_loss[3] += log_p_a.detach().mean()
            mean_loss[4] += q.detach().mean()
  
            if self.adaptatif:
                # print((-self.alpha*(log_p_a.detach().to(self.device) + self.seuil)).shape)
                l_alpha = torch.mean(-self.alpha*(log_p_a.detach().to(self.device) + self.seuil)) # log_p_a correct?
                
                self.optimizer_alpha.zero_grad()
                l_alpha.backward()
                self.optimizer_alpha.step()

                mean_loss[5] += l_alpha.detach()
            
            # update target networks
            for old_param, new_param in zip(self.critique_target_1.parameters(),self.critique_1.parameters()):
                old_param.data.copy_(self.p*old_param.data + (1-self.p)* new_param.data) 

            for old_param, new_param in zip(self.critique_target_2.parameters(),self.critique_2.parameters()):
                old_param.data.copy_(self.p*old_param.data + (1-self.p)* new_param.data) 

        return mean_loss[0]/self.nb_update, mean_loss[1]/self.nb_update, mean_loss[2]/self.nb_update,mean_loss[3]/self.nb_update, mean_loss[4]/self.nb_update, mean_loss[5]/self.nb_update, self.alpha.detach() if self.adaptatif else self.alpha


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
        if self.test:
            return False

        self.nbEvents+=1
        self.count += 1

        return self.count % self.freq_opti == 0 and self.count > self.random_sampling
    

    
