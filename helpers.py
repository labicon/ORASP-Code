import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition Defintion
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Replay Memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden=128):
        super(DQN, self).__init__()
        n_hidden = max(n_hidden, n_observations)
        self.fc1 = nn.Linear(n_observations, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def hierarchy_pos(G, root=None, width=20., vert_gap = 0.4, vert_loc = 0, xcenter = 0.5):  
    # From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):   
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos          
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# Generating Subassemblies in a recursive manner
def recurGen(p, H, G):
    currCon = list(H.edges())
    if(len(currCon) > 0):
        i = G.number_of_nodes() + 1
        for c in currCon:
            Hnew = H.copy()
            Hnew.remove_edges_from([c])
            if(c in seqConstraint.keys() and seqConstraint[c] in Hnew.edges()):
                pass # This means a feasability constraint has been failed!
            else:
                idString = str(Hnew.edges())
                nextState = next((x for x, y in nx.get_node_attributes(G,'idS').items() if y == idString), None)
                if(nextState is not None):
                    Rew, Hnew = R(p, c, H)
                    G.add_edge(p, nextState, a=c, r=Rew)
                else:
                    G.add_node(i, idS=idString)
                    Rew, Hnew = R(p,c,H)
                    G.add_edge(p, i, a=c, r=Rew)
                    G = recurGen(i, Hnew, G)
            i = G.number_of_nodes() + 1
        return G
    else:
        return G


# Getting next set of possible states and actions
def nextGen(H):
    currCon = list(H.edges())
    G = nx.DiGraph()
    G.add_node(1, value=H, idS=str(H.edges()))

    if(len(currCon) > 0):
        for c in currCon:
            i = G.number_of_nodes() + 1
            if(c in seqConstraint.keys() and seqConstraint[c] in Hnew.edges()):
                pass # This means a feasability constraint has been failed!
            else:
                Hnew = H.copy()
                Hnew.remove_edges_from([c])
                G.add_node(i, idS=str(Hnew.edges()))
                Rew, Hnew = R(1,c,H)
                G.add_edge(1, i, a=c, r=Rew)
        return G
    else:
        return None
    
    
# Checking Feasability
def T(sN, s, a):
    # Sequential Constraints are already handled via the tree generation!
    pass


# Cost structure which allows for an intermediary construction zone 
# between the supply vehicle and construction area
def Rcaz(s,a,H): # Reward Function Assuming there is a CAZ
    Rewards[(s,a)] = 0
    
    APieces = list(nx.connected_components(H))
    Hnew = H.copy()
    Hnew.remove_edges_from([a])
    BPieces = list(nx.connected_components(Hnew))
    if(Hnew.number_of_edges() == 0): # Check if fully deconstructed
        Rewards[(s,a)] = 1
    elif(len(BPieces) - len(APieces) > 0):
        diffPieces = [list(i) for i in BPieces if i not in APieces and len(i) <= 3]
        # Check if removing last connection for a given part
        for i in diffPieces:
            # Check if creating multiple assemblies and sizes of these new assemblies
            if(len(i) == 1):
                if(Hnew.nodes[i[0]]["loc"] == "CL"): # Going from CL to SV
                    Hnew.nodes[i[0]]["loc"] = "SV"
                    Rewards[(s,a)] += -(0.0468 + 0.0499) #SV-CL + CL-SV
                else: # Going from CAZ to SV
                    Hnew.nodes[i[0]]["loc"] = "SV"
                    Rewards[(s,a)] += -(0.0403 + 0.0420) #SV-CAZ + CAZ-SV

            elif(len(i) == 2): # Have to fix last bit problem
                for p in i:
                    if(Hnew.nodes[p]["loc"] == "CL"): # Going from CL to CAZ
                        Hnew.nodes[p]["loc"] = "CAZ"
                Rewards[(s,a)] += -(0.0307 + 0.0415) #SV-CL + CL-CAZ2

            elif(len(i) == 3): # Have to fix last problem
                for p in i:
                    if(Hnew.nodes[p]["loc"] == "CL"): # Going from CL to CAZ
                        Hnew.nodes[p]["loc"] = "CAZ"
                Rewards[(s,a)] += -(0.0307 + 0.0475) #SV-CL + CL-CAZ3
    return Rewards[(s,a)], Hnew


# Reward Function Assuming there is NO CAZ (i.e, structures are constructed at the Supply Vehicle)
def RNOcaz(s,a,H):
    s = str(s)
    Rewards[(s,a)] = 0
    
    APieces = list(nx.connected_components(H))
    Hnew = H.copy()
    Hnew.remove_edges_from([a])
    BPieces = list(nx.connected_components(Hnew))
    if(Hnew.number_of_edges() == 0): # Check if fully deconstructed
        Rewards[(s,a)] = 1
    elif(len(BPieces) - len(APieces) > 0):
        diffPieces = [list(i) for i in BPieces if i not in APieces and len(i) <= 3]
        # Check if removing last connection for a given part
        for i in diffPieces:
            # Check if creating multiple assemblies and sizes of these new assemblies
            if(len(i) == 1):
                Rewards[(s,a)] += -(0.0468 + 0.0499) #SV-CL + CL-SV

            elif(len(i) == 2):
                Rewards[(s,a)] += -(0.0749 + 0.0499) #SV-CL + CL-SV2

            elif(len(i) == 3):
                Rewards[(s,a)] += -(0.0869 + 0.0499) #SV-CL + CL-SV3
    return Rewards[(s,a)], Hnew



def Rcustom(s,a,H):
    s = str(s)
    Rewards[(s,a)] = -0.1
    
    APieces = list(nx.connected_components(H))
    Hnew = H.copy()
    Hnew.remove_edges_from([a])
    BPieces = list(nx.connected_components(Hnew))
    if(Hnew.number_of_edges() == 0): # Check if fully deconstructed
        Rewards[(s,a)] = 1
    elif(len(BPieces) - len(APieces) > 0):
        diffPieces = [list(i) for i in BPieces if i not in APieces and len(i) <= 3]
        # Check if removing last connection for a given part
        for i in diffPieces:
            # Check if creating multiple assemblies and sizes of these new assemblies
            if(len(i) == 1):
                Rewards[(s,a)] += -1 #SV-CL + CL-SV

            elif(len(i) == 2):
                Rewards[(s,a)] += -1.5 #SV-CL + CL-SV2

            elif(len(i) == 3):
                Rewards[(s,a)] += -1.75 #SV-CL + CL-SV3
    return Rewards[(s,a)], Hnew


def Rsimple(s,a,H):
    lab = str(s)
    Hnew = H.copy()
    Hnew.remove_edges_from([a])
    edges = fullE.copy()
    if(edges.index(a) < H.number_of_edges()-1):
        Rewards[(lab,a)] = H.number_of_edges() - edges.index(a)
    else:
        Rewards[(lab,a)] = -1
    return Rewards[(lab,a)], Hnew



# Allows you to pick which reward function to use!
def R(s, a, H):
    lab = str(s)
    if((lab,a) not in Rewards.keys()):
        Rewards[(lab,a)], Hnew = RNOcaz(s,a,H)
    else:
        Hnew = H.copy()
        Hnew.remove_edges_from([a])
    return Rewards[(lab,a)], Hnew