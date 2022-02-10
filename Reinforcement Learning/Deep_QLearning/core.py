import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# classe utile pour ajouter du bruit (utile par exemple pour DDPG)
class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return torch.FloatTensor(self.X)


#################### Extracteurs de Features à partir des observations ##################################"

# classe abstraite générique
class FeatureExtractor(object):
    def __init__(self):
        super().__init__()

    def getFeatures(self,obs):
        pass

# Ne fait rien, conserve les observations telles quelles
# A utiliser pour CartPole et LunarLander
class NothingToDo(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        ob=env.reset()
        ob=ob.reshape(-1)
        self.outSize=len(ob)

    def getFeatures(self,obs):
        #print(obs)
        return obs.reshape(1,-1)

# Ajoute le numero d'iteration (a priori pas vraiment utile et peut destabiliser dans la plupart des cas etudiés)
class AddTime(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        ob=env.reset()
        ob=ob.reshape(-1)
        self.env=env
        self.maxTime=env.config["duration"]
        self.outSize=len(ob)+1

    def getFeatures(self,obs):
        #print(obs)
        return np.concatenate((obs.reshape(1,-1),np.array([self.env.steps/self.maxTime]).reshape(1,-1)),axis=1)

######  Pour Gridworld #############################"

class MapFromDumpExtractor(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize

    def getFeatures(self, obs):
        #prs(obs)
        return obs.reshape(1,-1)


# Representation simplifiée, pas besoin d'encoder les murs et les etats terminaux qui ne bougent pas
# Extracteur recommandé pour Gridworld pour la plupart des algos
class MapFromDumpExtractor2(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize=env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize*3

    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1,-1)


# Representation simplifiée, avec position agent
class MapFromDumpExtractor3(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize=env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize*2+2

    def getFeatures(self, obs):
        state=np.zeros((2,np.shape(obs)[0],np.shape(obs)[1]))
        state[0] = np.where(obs == 4, 1, state[0])
        state[1] = np.where(obs == 6, 1, state[1])
        pos=np.where(obs==2)
        posx=pos[0]
        posy=pos[1]
        return np.concatenate((posx.reshape(1,-1),posy.reshape(1,-1),state.reshape(1,-1)),axis=1)

# Representation (très) simplifiée, uniquement la position de l'agent
# Ne permet pas de gérer les éléments jaunes et roses de GridWorld
class MapFromDumpExtractor4(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        self.outSize=2

    def getFeatures(self, obs):
        pos=np.where(obs==2)
        posx=pos[0]
        posy=pos[1]
        #print(posx,posy)
        return np.concatenate((posx.reshape(1,-1),posy.reshape(1,-1)),axis=1)

# Representation simplifiée, pour conv
class MapFromDumpExtractor5(FeatureExtractor):
    def __init__(self,env):
        super().__init__()

        self.outSize=(3,env.start_grid_map.shape[0],env.start_grid_map.shape[1])

    def getFeatures(self, obs):
        state=np.zeros((1,3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0,0]=np.where(obs == 2,1,state[0,0])
        state[0,1] = np.where(obs == 4, 1, state[0,1])
        state[0,2] = np.where(obs == 6, 1, state[0,2])
        return state


# Autre possibilité de représentation, en terme de distances dans la carte
class DistsFromStates(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        self.outSize=16

    def getFeatures(self, obs):
        #prs(obs)
        #x=np.loads(obs)
        x=obs
        #print(x)
        astate = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(x == 2)
        ))
        astate=np.array(astate)
        a3=np.where(x == 3)
        d3=np.array([0])
        if len(a3[0])>0:
            astate3 = np.concatenate(a3).reshape(2,-1).T
            d3=np.power(astate-astate3,2).sum(1).min().reshape(1)

            #d3 = np.array(d3).reshape(1)
        a4 = np.where(x == 4)
        d4 = np.array([0])
        if len(a4[0]) > 0:
            astate4 = np.concatenate(a4).reshape(2,-1).T
            d4 = np.power(astate - astate4, 2).sum(1).min().reshape(1)
            #d4 = np.array(d4)
        a5 = np.where(x == 5)
        d5 = np.array([0])
        #prs(a5)
        if len(a5[0]) > 0:
            astate5 = np.concatenate(a5).reshape(2,-1).T
            d5 = np.power(astate - astate5, 2).sum(1).min().reshape(1)
            #d5 = np.array(d5)
        a6 = np.where(x == 6)
        d6 = np.array([0])
        if len(a6[0]) > 0:
            astate6 = np.concatenate(a6).reshape(2,-1).T
            d6 = np.power(astate - astate6, 2).sum(1).min().reshape(1)
            #d6=np.array(d6)

        #prs("::",d3,d4,d5,d6)
        ret=np.concatenate((d3,d4,d5,d6)).reshape(1,-1)
        ret=np.dot(ret.T,ret)
        return ret.reshape(1,-1)

#######################################################################################





# Si on veut travailler avec des CNNs plutôt que des NN classiques (nécessite une entrée adaptée)
class convMDP(nn.Module):
    def __init__(self, inSize, outSize, layers=[], convs=None, finalActivation=None, batchNorm=False,init_batchNorm=False,activation=torch.tanh,dropout=0.0, maxPool=None):
        super(convMDP, self).__init__()
        #print(inSize,outSize)

        self.inSize=inSize
        self.outSize=outSize
        self.batchNorm=batchNorm
        self.init_batchNorm = init_batchNorm
        self.activation=activation
        self.dropout=None
        if dropout>0:
            self.dropout=torch.nn.Dropout(dropout)
        print("inSize:",inSize)
        self.convs=None
        if convs is not None:
            self.convs = nn.ModuleList([])
            nbChans,hSize,wSize=inSize
            for x in convs:
                if nbChans!=x[0]:
                    raise RuntimeError("incompatible numbers of channels : ", x[0], nbChans)
                self.convs.append(nn.Conv2d(x[0], x[1], x[2], stride=x[3]))
                hSize=((hSize-x[2])/x[3])+1
                wSize = ((wSize - x[2]) / x[3]) + 1
                nbChans = x[1]
                if maxPool is not None:
                    self.convs.append(nn.MaxPool2d(maxPool))
                    hSize = (hSize - maxPool)  + 1
                    wSize = (wSize - maxPool)  + 1
                print(nbChans,hSize,wSize)
            inSize=int(nbChans*hSize*wSize)

        #print(inSize)

        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        i=0
        if batchNorm or init_batchNorm:
            self.bn.append(nn.BatchNorm1d(num_features=inSize))
        #print(inSize)
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            if batchNorm:
                self.bn.append(nn.BatchNorm1d(num_features=x))

            #nn.init.xavier_uniform_(self.layers[i].weight)
            nn.init.normal_(self.layers[i].weight.data, 0.0, 0.02)
            nn.init.normal_(self.layers[i].bias.data,0.0,0.02)
            i+=1
            inSize = x


        self.layers.append(nn.Linear(inSize, outSize))

        #nn.init.uniform_(self.layers[-1].weight)
        nn.init.normal_(self.layers[-1].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layers[-1].bias.data, 0.0, 0.02)
        self.finalActivation=finalActivation

    def setcuda(self, device):
        self.cuda(device=device)



    def forward(self, x):

        if self.convs is not None:
            x = x.view(-1, *self.inSize)
            n=x.size()[0]
            i=0
            for c in self.convs:
                x=c(x)
                x=self.activation(x)
                i+=1
            x=x.view(n,-1)


        if self.batchNorm or self.init_batchNorm:
            x=self.bn[0](x)

        x = self.layers[0](x)

        for i in range(1, len(self.layers)):
            x = self.activation(x)
            if self.dropout is not None:
                x=self.dropout(x)
            #if self.drop is not None:
            #    x = nn.drop(x)
            if self.batchNorm:
                x = self.bn[i](x)
            x = self.layers[i](x)


        if self.finalActivation is not None:
            x=self.finalActivation(x)
        #print("f",x.size())
        return x


# Classe basique de NN générique
# Accepte une liste de tailles de couches pour la variables layers (permet de définir la structure)
class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[], finalActivation=None, activation=torch.tanh,dropout=0.0):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        self.activation = activation
        self.finalActivation = finalActivation
        self.dropout = None
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)

    def setcuda(self, device):
        self.cuda(device=device)

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = self.activation(x)
            if self.dropout is not None:
                x=self.dropout(x)

            x = self.layers[i](x)

        if self.finalActivation is not None:
            x=self.finalActivation(x)

        return x