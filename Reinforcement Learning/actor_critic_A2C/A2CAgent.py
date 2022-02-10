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
import torch
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from memory import *
import copy
import torch.nn as nn

class A2CAgent(object):

    def __init__(self, env, opt, discount = 0.99,\
                 learning_rate = 1e-4, freq_opti = 10000,\
                sample_size = 10,lbda = 1,entropy = 0):
        """
        discount : discount reward
        freq_opti : number of optimisation to update the action model
        sample_size : number of iteration before train
        entropy: entropy coef to add (if 0 nothing is added)
        """
        self.opt=opt
        self.env=env

        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents = 0
        self.nbEventsLastUpdate = 0
        self.nbUpdates = 0
        

        self.replay = [[]]
        self.cumulated_rewards = []
        


        # parameters specific to A2C
        self.lbda = opt.get("lbda",lbda)
        self.freq_opti = opt.get("freqOptim",freq_opti)
        self.gamma = opt.get("discount",discount)
        self.sample_size = opt.get("sample_size",sample_size)
        learning_rate = opt.get("learning_rate",learning_rate)
        self.entropy = opt.get("entropy",entropy)
        self.dkl = torch.nn.KLDivLoss()

        n_in = self.featureExtractor.outSize
        n_out = self.env.action_space.n
        
        self.politique = NN(n_in, n_out, layers=[30,30], finalActivation=nn.Softmax(dim=1), \
                            activation=torch.tanh,dropout=0.0)
        
        self.critique = NN(n_in, 1, layers=[30,30], finalActivation=None, \
                            activation=torch.tanh,dropout=0.0)

        self.critique_target = copy.deepcopy(self.critique)
        self.count = 0
        
        self.loss = torch.nn.HuberLoss() #Pour l'apprentissage
        self.optimizer_politique = torch.optim.Adam(self.politique.parameters(), learning_rate)
        self.optimizer_critique = torch.optim.Adam(self.critique.parameters(), learning_rate)


    def act(self, obs):
        proba = self.politique(torch.Tensor(obs))
        return int(torch.multinomial(proba, num_samples=1))

    # sauvegarde du modèle
    def save(self,outputDir):
        torch.save(self.critique.state_dict(),outputDir+"_critique")
        torch.save(self.politique.state_dict(),outputDir+"_politique")

    # chargement du modèle.
    def load(self,inputDir):
        self.critique.load_state_dict(torch.load(inputDir+"_critique"))
        self.politique.load_state_dict(torch.load(inputDir+"_politique"))


    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return
        self.count += 1
        
        
	
        l_politique_tot = 0
        l_critique_tot = 0
        mean_cible = 0

        nb_traj = len(self.replay) - 1
        
        traj = np.concatenate(np.array(self.replay[:-1]))
        all_obs = torch.Tensor(np.concatenate(traj[:,0]))
        old_p = self.politique(all_obs).detach()
	
        # learn critique
        self.optimizer_critique.zero_grad()
        for i in range(len(self.replay)-1): # for each trajectory

            traj = np.array(self.replay[i])
            obs     = torch.Tensor(np.concatenate(traj[:,0]))

            rewards = torch.Tensor(list(traj[:,2])).unsqueeze(1) # self.cumulated_rewards[i]
            
            new_obs = torch.Tensor(np.concatenate(traj[:,3]))
            #On optimise le critique V

            y_hat = self.critique(obs)
            y = rewards + self.gamma*self.critique_target(new_obs)
            
            y = y.detach()
        
            l_critique = self.loss(y_hat, y)
            l_critique.backward()

            with torch.no_grad():
                l_critique_tot += l_critique
                mean_cible += torch.mean(y)
        
        self.optimizer_critique.step()
        
        
        # As = self.get_GAE()

        # learn actor
        self.optimizer_politique.zero_grad()
        for i in range(len(self.replay)-1):

            #Evalue A 
                #A = rewards\
            # taille_traj = len(self.replay[i])
            traj = np.array(self.replay[i])
            
            new_obs = torch.Tensor(np.concatenate(traj[:,3]))
            obs     = torch.Tensor(np.concatenate(traj[:,0]))
            actions = torch.tensor(np.array(traj[:,1],dtype = np.int))

            A = self.cumulated_rewards[i].squeeze(1).detach()

            #Evalue J
            

            pol = self.politique(obs)
            l_politique = - torch.sum(torch.log(pol[torch.arange(pol.size(0)),actions])*A) 
            if self.entropy > 0:
                l_politique -= self.entropy * torch.sum(pol*torch.log(pol))
               
            l_politique.backward()

            with torch.no_grad():
                l_politique_tot += l_politique/len(traj)
            
        
        
        self.optimizer_politique.step()
        
        # reinitialisation
        self.replay = [[]]
        self.cumulated_rewards = []

        if self.count % self.freq_opti == 0:
            self.critique_target = copy.deepcopy(self.critique)
        
        self.nbEventsLastUpdate = 0

        
        
        new_p = self.politique(all_obs).detach()
        DKL = self.dkl(new_p.log(),old_p).detach()


        return float(l_politique_tot)/nb_traj,float(l_critique_tot)/ nb_traj, mean_cible/nb_traj, DKL


    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

           
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = [ob, action, reward, new_ob, done]
            
            self.replay[-1] += [tr] # add action to last trajectory

            if done or it == self.opt.maxLengthTrain:  # if end
                # cumulated reward
                with torch.no_grad(): 
                    # update last trajectory
                    last_traj = self.replay[-1]
                    cumulated_reward = torch.zeros(len(last_traj)) # depend of TD0, TD1 or TDlambda
                    
                    for t in reversed(range(len(last_traj))):
                        next_state = torch.tensor(last_traj[t][3][0])
                        cur_state = torch.tensor(last_traj[t][0][0])
                        Vcur = self.critique_target(cur_state)[0]

                        cumulated_reward[t] = last_traj[t][2] - Vcur # reward
                        cumulated_reward[t] += self.gamma*self.critique_target(next_state)[0]# add next reward
                        if t != len(last_traj) - 1:
                            cumulated_reward[t] += self.gamma*self.lbda*cumulated_reward[t+1] # add cumulation (before is same with lambda = 0)
                        
                    self.cumulated_rewards += [cumulated_reward.unsqueeze(1)] # add to buffer

                
                self.replay += [[]] # add empty trajectory to buffer

    '''
    def get_GAE(self):
        """
            return the Generalized Advantage Estimation for data in the buffer
        """
        advs_replay = []
        with torch.no_grad():
            for traj in self.replay[:-1]: # for each trajectory
                advs_replay += [torch.zeros(len(traj))] # add to list
                
                for t in reversed(range(len(traj))):

                    current = traj[t][0]
                    next = traj[t][3]
                    Vnext = self.critique_target(torch.tensor(next,dtype=torch.float))[0][0]
                    
                    cur_reward = traj[t][2]

                    # cur_reward + gamma*Vnext - Vcur  
                    delta = -Vcur + cur_reward + self.gamma * Vnext
                        
                            
                    advs_replay[-1][t] = delta
                    
                    if t != len(traj) - 1:
                        advs_replay[-1][t]+=(self.gamma * self.lbda * advs_replay[-1][t+1]) # gamma and lambda from next actions
        return advs_replay
    '''
                    
                    
    
    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False

        self.nbEvents+=1
        self.nbEventsLastUpdate+=1
        return self.nbEventsLastUpdate >= self.sample_size and done
    

