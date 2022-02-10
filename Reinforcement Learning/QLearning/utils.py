import time
import subprocess
from collections import namedtuple,defaultdict
import logging
import json
import os
import yaml
import gym
import sys
import threading
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def loadTensorBoard(outdir):
    t = threading.Thread(target=launchTensorBoard, args=([outdir]))
    t.start()

def launchTensorBoard(tensorBoardPath):
    print('tensorboard --logdir=' + tensorBoardPath)
    ret=os.system('tensorboard --logdir='  + tensorBoardPath)
    if ret!=0:
        syspath = os.path.dirname(sys.executable)
        print(os.path.dirname(sys.executable))
        ret = os.system(syspath+"/"+'tensorboard --logdir=' + tensorBoardPath)
    return



class LogMe(dict):
    def __init__(self,writer,term=True):
        self.writer = writer
        self.dic = defaultdict(list)
        self.term = term
    def write(self,i):
        if len(self.dic)==0: return
        s=f"Epoch {i} : "
        for k,v in self.dic.items():
            self.writer.add_scalar(k,sum(v)*1./len(v),i)
            s+=f"{k}:{sum(v)*1./len(v)} -- "
        self.dic.clear()
        if self.term: logging.info(s)
    def update(self,l):
        for k,v in l:
            self.add(k,v)
    def direct_write(self,k,v,i):
        self.writer.add_scalar(k,v,i)
    def add(self,k,v):
        self.dic[k].append(v)

def save_src(path):
    current_dir = os.getcwd()
    package_dir = current_dir.split('RL', 1)[0]
    #path = os.path.abspath(path)
    os.chdir(package_dir)
    #print(package_dir)
    src_files = subprocess.Popen(('find', 'RL', '-name', '*.py', '-o', '-name', '*.yaml'),
                                 stdout=subprocess.PIPE)
    #print(package_dir,path)
    #path=os.path.abspath(path)


    #print(str(src_files))

    subprocess.check_output(('tar', '-zcf', path+"/arch.tar", '-T', '-'), stdin=src_files.stdout, stderr=subprocess.STDOUT)
    src_files.wait()
    os.chdir(current_dir)



def prs(*args):
    st = ""
    for s in args:
        st += str(s)
    print(st)


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream,Loader=yaml.Loader)
    return DotDict(opt)

def write_yaml(file,dotdict):
    d=dict(dotdict)
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)

global verbose
verbose=2

def printv(*o,p=0):
    if p<verbose:
        print(*o)

def checkConfUpdate(outdir,config):
    if os.path.exists(os.path.join(outdir, 'update.yaml')):
        try:
            config2 = load_yaml(os.path.join(outdir, 'update.yaml'))
            print("update conf with:",config2)
            for k in config2:
                config[k] = config2[k]
            if config2.get("execute") is not None:
                exec(config2["execute"])
            now = datetime.now()
            date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
            write_yaml(os.path.join(outdir, 'newConfig_' + date_time + '.yaml'), config)
            os.remove(os.path.join(outdir, 'update.yaml'))
        except yaml.scanner.ScannerError:
            print("update config failed, yaml error")
        except SyntaxError:
            print("pb with exec code in config")

def logConfig(logger,config):
    print(str(yaml.dump(dict(config))))
    st = ""
    for i, v in dict(config).items():
        st += "\t \t \t \n" + str(i) + ":" + str(v)
    logger.writer.add_text("config", st, 1)

def logRun(name,config,agent_object):
    global agent
    agent=agent_object
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
    outdir = "./XP/" + config["env"] + "/"+name+"_" + date_time
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    config["commit"] = subprocess.check_output(['git', 'show-ref']).decode('utf-8')
    write_yaml(os.path.join(outdir, 'config.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)
    logConfig(logger, config)
    return logger,outdir


def init(config_file, algoName):
    config = load_yaml(config_file)
    env = gym.make(config["env"])
    if config.get("import") is not None:
        exec(config["import"])

    if config.get("execute") is not None:
        exec(config["execute"])

    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y-%HH%M-%SS")
    outdir = "./XP/" + config["env"] + "/" + algoName + "_" + date_time

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'config.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    return env, config, outdir, logger