import gym
import sys
import os
import time
import copy
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
#from PIL import Image as Image
import matplotlib.pyplot as plt
from gym.envs.toy_text import discrete
from itertools import groupby
from operator import itemgetter
from contextlib import closing
from six import StringIO, b

# define colors
# 0: black; 1 : gray; 2 : blue; 3 : green; 4 : red

COLORS = {0:[0,0,0], 1:[128,128,128], \
          2:[0,0,255], 3:[0,255,0], \
          5:[255,0,0], 6:[255,0,255], \
          4:[255,255,0]}
COLORSDIC = {0: "white", 1:"gray", 2:"blue",3:"green",4:"cyan",5:"red",6:"magenta"} 

def str_color(s):
    return utils.colorize(" ",COLORSDIC[int(s)],highlight=True)
   
class GridworldEnv(discrete.DiscreteEnv):
    """ Environnement de Gridworld 2D avec le codage suivant : 
            0: case vide
            1: mur
            2: joueur
            3: sortie
            4: objet a ramasser
            5: piege mortel
            6: piege non mortel
        actions : 
            0: South
            1: North
            2: West
            3: East
    """

    metadata = {
        'render.modes': ['human', 'ansi','rgb_array'], #, 'state_pixels'],
        'video.frames_per_second': 1
    }
    num_env = 0
    plan='gridworldPlans/plan0.txt'
    rewards = {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}

    def __init__(self):
        self._make(GridworldEnv.plan,rewards=GridworldEnv.rewards)
    
    def setPlan(self,plan,rew=None):
        if rew is None:
            rew=GridworldEnv.rewards
        self._make(plan,rew)
    @staticmethod
    def state2str(state):
        return str(state.tolist())
    @staticmethod
    def str2state(s):
        return np.array(eval(s))

    #def randomGenMapAndGoal(self,size,wallDensity,fireDensity,):



    def getStateFromObs(self,obs):
        states,p=self.getMDP()
        return self.states[GridworldEnv.state2str(obs)]

    def _make(self,plan,rewards):
        self.rewards=rewards
        self.actions={0:[1,0],1:[-1,0],2:[0,-1],3:[0,1]}
        self.nA = len(self.actions)
        self.nbMaxSteps=1000
        self.action_space = spaces.Discrete(self.nA)
        if not os.path.exists(plan):
            this_file_path = os.path.dirname(os.path.realpath(__file__))
            self.grid_map_path = os.path.join(this_file_path, plan)
        else:
            self.grid_map_path=plan
        self.obs_shape = [128, 128, 3]
        self.start_grid_map, self.goals = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = np.copy(self.start_grid_map)  # current grid map
        self.nbSteps=0
        self.rstates = {}
        self.P=None
        self.nS=None
        self.startPos=self._get_agent_pos(self.current_grid_map)
        self.currentPos=copy.deepcopy(self.startPos)
        GridworldEnv.num_env += 1
        self.this_fig_num = GridworldEnv.num_env
        self.lastaction = None
        self.observation_space = None
        
    def getMDP(self):
        if self.P is None:
            self.P={}
            self.states={self.state2str(self.start_grid_map):0}
            self._getMDP(self.start_grid_map, self.startPos)
            self.nS = len(self.states)
            self.observation_space = spaces.Discrete(self.nS)
        tabstates=[""]*len(self.states)
        for a,v in self.states.items():
            tabstates[v]=a
        return (tabstates,self.P)



    def _getMDP(self,gridmap,state):
        cur = self.states[self.state2str(gridmap)]
        #cur = 0
        succs={0:[],1:[],2:[],3:[]}
        self.P[cur]=succs
        self._exploreDir(gridmap,state,[1,0],0,2,3)
        self._exploreDir(gridmap, state, [-1, 0], 1, 2, 3)
        self._exploreDir(gridmap, state, [0, 1], 3, 0, 1)
        self._exploreDir(gridmap, state, [0, -1], 2, 0, 1)


    def _exploreDir(self,gridmap,state,dir,a,b,c):
        cur=self.states[self.state2str(gridmap)]
        #print(cur)
        gridmap = copy.deepcopy(gridmap)
        succs=self.P[cur]
        nstate = copy.deepcopy(state)
        nstate[0]+=dir[0]
        nstate[1] += dir[1]

        if nstate[0]<gridmap.shape[0] and nstate[0]>=0 and nstate[1]<gridmap.shape[1] and nstate[1]>=0 and gridmap[nstate[0],nstate[1]]!=1:
                oldc=gridmap[nstate[0],nstate[1]]
                gridmap[state[0],state[1]] = 0
                gridmap[nstate[0],nstate[1]] = 2
                ng=self.state2str(gridmap)
                done = (oldc == 3 or oldc == 5)
                if ng in self.states:
                    ns=self.states[ng]
                else:
                    ns=len(self.states)
                    self.states[ng]=ns
                    if not done:
                        self._getMDP(gridmap,nstate)
                r=self.rewards[oldc]

                succs[a].append((0.8, self.states[ng],r,done))
                succs[b].append((0.1, self.states[ng], r, done))
                succs[c].append((0.1, self.states[ng], r, done))
        else:
            succs[a].append((0.8,cur,self.rewards[0],False))
            succs[b].append((0.1, cur, self.rewards[0], False))
            succs[c].append((0.1, cur, self.rewards[0], False))




    def _get_agent_pos(self, grid_map):
        state = list(map(
                 lambda x:x[0] if len(x) > 0 else None,
                 np.where(grid_map == 2)
             ))
        return state


    def step(self, action):
        self.nbSteps += 1
        c = self.start_grid_map[self.currentPos[0],self.currentPos[1]]
        if c==3 or c==5 : ## Done == True au coup d'avant
            return self.current_grid_map,0,self.done,{}
        action = int(action)
        p = np.random.rand()
        if p<0.2:
            p = np.random.rand()
            if action==0 or action==1:
                if p < 0.5:
                    action=2
                else:
                    action=3
            else:
                if p < 0.5:
                    action=0
                else:
                    action=1
        npos = (self.currentPos[0] + self.actions[action][0], self.currentPos[1] + self.actions[action][1])
        rr=-1*(self.nbSteps>self.nbMaxSteps)
        if npos[0] >= self.current_grid_map.shape[0] or npos[0] < 0 or npos[1] >= self.current_grid_map.shape[1] or npos[1] < 0 or self.current_grid_map[npos[0],npos[1]]==1:
            return (self.current_grid_map, self.rewards[0]+rr, self.nbSteps>self.nbMaxSteps, {})
        c=self.current_grid_map[npos]
        r = self.rewards[c]+rr
        self.done=(c == 3 or c == 5 or self.nbSteps>self.nbMaxSteps)
        self.current_grid_map[self.currentPos[0],self.currentPos[1]] = 0
        self.current_grid_map[npos[0],npos[1]] = 2
        self.currentPos = npos
        self.lastaction = action
        return (self.current_grid_map,r,self.done,{})

    def reset(self):
        self.currentPos = copy.deepcopy(self.startPos)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.nbSteps=0
        self.lastaction = None
        self.done=False
        return self.current_grid_map

    def _read_grid_map_old(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            grid_map = f.readlines()
        grid_map_array = np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grid_map
            ))
        )
        return grid_map_array



    def _gridmap_to_img(self, grid_map, obs_shape=None):
        if obs_shape is None:
            obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.uint8)
        gs0 = int(observation.shape[0] / grid_map.shape[0])
        gs1 = int(observation.shape[1] / grid_map.shape[1])
        for i in range(grid_map.shape[0]):
            for j in range(grid_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[grid_map[i, j]])
        return observation

    def render(self, pause=0.00001, mode='rgb_array', close=False):
        if mode =='human' or mode =='ansi':
            outfile = StringIO() if mode == 'ansi' else sys.stdout
            desc = self.current_grid_map.tolist()
            desc = [[str_color(c) for c in line] for line in desc]
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(["South","North","West","East"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")
            if mode != 'human':
                with closing(outfile):
                    return outfile.getvalue()            
            return
        img = self._gridmap_to_img(self.current_grid_map)
        fig = plt.figure(self.this_fig_num)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()
        if pause>0:
            plt.pause(pause)
        return img

    def _close_env(self):
        plt.close(self.this_fig_num)
        return

    def close(self):
        super(GridworldEnv,self).close()
        self._close_env()
    def changeState(self,gridmap):
        self.current_grid_map=gridmap
        self.currentPos=self._get_agent_pos(gridmap)

    # pour reset en mode RL drivé par un but
    # points rose pas gérés
    def resetStartAndGoalState_old(self, goal=None):

        shape = self.start_grid_map.shape
        if goal is None:
            goal = self.start_grid_map.reshape(-1).copy()
            start = goal.copy()
            goal = np.where(goal == 2, 0, goal)  # on retire l'agent de sa position initiale pour former le but
            goal = np.where(goal == 4, 0, goal)  # on retire les éléments qu'il est censé manger pour former le but
            goals = np.where(goal == 3)[0]
            if len(goals) == 0:
                raise RuntimeError("No goal defined")

            if len(goals) > 1:
                goals = np.random.choice(goals, 1)  # si plusieurs buts finaux, on choisit le but au hasard parmi eux
            goal[goals] = 2
        else:
            goals= np.where(goal == 2)[0]
            if len(goals) > 1:
                raise RuntimeError("Too many goals defined")
            if goal.shape!=shape:
                raise RuntimeError("Goal has not the correct shape")
        start = np.where(start == 3, 0, start)  # On retire les buts de la carte de depart
        #start[goals] = 3  # On remet le but choisi
        self.currentPos = copy.deepcopy(self.startPos)
        self.current_grid_map = start.reshape(shape)
        self.nbSteps = 0
        self.lastaction = None
        self.done = False

        return self.current_grid_map, goal.reshape(shape)



    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as f:
            l = f.readlines()
        grids=[list(g) for k, g in groupby(l, lambda x: x[0] != "#") if k]
        grids = [np.array(
            list(map(
                lambda x: list(map(
                    lambda y: int(y),
                    x.split(' ')
                )),
                grids[i]
            ))
        ) for i in range(len(grids))]
        grid_map_array = grids[0]
        goals=grids[1:]

        return grid_map_array, goals

    def sampleGoal(self):
        if self.goals is None:
            raise RuntimeError("No goal Defined")

        l=np.arange(0,len(self.goals))
        i=np.random.choice(l,1)[0]
        return self.goals[i],self._gridmap_to_img(self.goals[i])