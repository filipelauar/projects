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

class PPOAgent_2(object):

    def __init__(self, env, opt, discount = 0.99,\
                 learning_rate = 1e-4, freq_opti = 10000,\
                sample_size = 10,lbda = 1, K = 1,epsilon = 0.2, entropy = 0):
        """
        discount : discount reward
        freq_opti : number of optimisation to update the action model
        sample_size : number of iteration in a sample
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        


        self.B = 0
        self.lbda = opt.get("lbda",lbda)
        self.freq_opti = opt.get("freqOptim",freq_opti)
        self.gamma = opt.get("discount",discount)
        self.sample_size = opt.get("sample_size",sample_size)
        self.K = opt.get("K",K)
        learning_rate = opt.get("learning_rate",learning_rate)
        self.epsilon = opt.get("epsilon",epsilon)
        self.entropy = opt.get("entropy",entropy)

        n_in = self.featureExtractor.outSize
        n_out = self.env.action_space.n
        
        self.politique = NN(n_in, n_out, layers=[30], finalActivation=nn.Softmax(dim = 1), \
                            activation=torch.tanh,dropout=0.0).to(self.device)
        
        self.critique = NN(n_in, 1, layers=[30], finalActivation=None, \
                            activation=torch.tanh,dropout=0.0).to(self.device)

        self.critique_target = copy.deepcopy(self.critique).to(self.device)
        self.count = 0
        
        self.loss = torch.nn.HuberLoss() #Pour l'apprentissage
        self.dkl = torch.nn.KLDivLoss()
        self.optimizer_politique = torch.optim.Adam(self.politique.parameters(), learning_rate)
        self.optimizer_critique = torch.optim.Adam(self.critique.parameters(), learning_rate)
        


    def act(self, obs):
        proba = self.politique(torch.Tensor(obs).to(self.device)).detach()
        return int(torch.multinomial(proba, num_samples=1))

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
        self.count += 1

        l_critique_tot = 0
        l_politique_tot = 0
        
        nb_traj = len(self.replay) - 1

        old_model = copy.deepcopy(self.politique)
        
        
        traj = np.concatenate(np.array(self.replay[:-1]))
            
        A = torch.cat(self.cumulated_rewards).squeeze(1).detach()


        new_obs = torch.Tensor(np.concatenate(traj[:,3]))
        obs     = torch.Tensor(np.concatenate(traj[:,0])).to(self.device)
        actions = torch.tensor(np.array(traj[:,1],dtype = np.int))
        with torch.no_grad():
            old_p = old_model(obs)
            
        for _ in range(self.K):
            self.optimizer_politique.zero_grad()
            new_p = self.politique(obs)
            range_traj = torch.arange(len(traj))
            coef = new_p[range_traj,actions]/old_p[range_traj,actions]

            l_politique = -torch.mean(torch.min(coef*A,torch.clip(coef,1-self.epsilon,1+self.epsilon)*A))
            if self.entropy > 0:
                l_politique += self.entropy * torch.sum(new_p*torch.log(new_p))
            l_politique.backward()
            self.optimizer_politique.step()
            with torch.no_grad():
                l_politique_tot += l_politique
        DKL = self.dkl(self.politique(obs).log(),old_p.log()).detach()

        self.optimizer_critique.zero_grad()
        for i in range(len(self.replay)-1):
            #Apprentissage critique
            traj = np.array(self.replay[i])

            obs     = torch.Tensor(np.concatenate(traj[:,0])).to(self.device)

            #On optimise le critique V
            y = self.cumulated_rewards[i].detach()
        
            y_hat = self.critique(obs)
            l_critique = self.loss(y_hat, y)
            l_critique.backward()

            with torch.no_grad():
                l_critique_tot += l_critique
        
        self.optimizer_critique.step()

        self.replay = [[]]
        self.cumulated_rewards = []

        if self.count % self.freq_opti == 0:
            self.critique_target = copy.deepcopy(self.critique)
        
        self.nbEventsLastUpdate = 0
        

        return float(l_politique)/self.K,float(l_critique_tot)/nb_traj, torch.mean(A), 0, DKL


    def get_new_cummulated_rewards(self):
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
                    Vcur = self.critique_target(torch.tensor(current,dtype=torch.float).to(self.device))[0][0]
                    Vnext = self.critique_target(torch.tensor(next,dtype=torch.float).to(self.device))[0][0]
                    
                    cur_reward = traj[t][2]

                    # cur_reward + gamma*Vnext - Vcur  
                    delta = -Vcur + cur_reward + self.gamma * Vnext
                        
                            
                    advs_replay[-1][t] = delta
                    
                    if t != len(traj) - 1:
                        advs_replay[-1][t]+=(self.gamma * self.lbda * advs_replay[-1][t+1]) # gamma and lambda from next actions
        return advs_replay

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
                        next_state = torch.tensor(last_traj[t][3][0]).to(self.device)
                        cur_state = torch.tensor(last_traj[t][0][0]).to(self.device)
                        Vcur = self.critique_target(cur_state)[0]

                        cumulated_reward[t] = last_traj[t][2] - Vcur # reward
                        cumulated_reward[t] += self.gamma*self.critique_target(next_state)[0]# add next reward
                        if t != len(last_traj) - 1:
                            cumulated_reward[t] += self.gamma*self.lbda*cumulated_reward[t+1] # add cumulation (before is same with lambda = 0)
                        
                    self.cumulated_rewards += [cumulated_reward.unsqueeze(1)] # add to buffer

                
                self.replay += [[]] # add empty trajectory to buffer
                    
                    
    
    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False

        self.nbEvents+=1
        self.nbEventsLastUpdate+=1
        return self.nbEventsLastUpdate >= self.sample_size and done
    

