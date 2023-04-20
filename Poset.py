import math

import ArcConFast as ArcCon
import networkx as nx

import Structures
import pptester
import pickle
import threading
from multiprocessing import Process,Manager
import time
import signal
import Identities
import concurrent.futures
import copy
import random
import ast
import itertools


TSAllN = Identities.Identity([],'TSAllN')

class Construction:
    def __init__(self,Name):
        pass

    def apply(self,G):
        return G

    def isApplicable(self,G):
        return True


class ConstructionSimplePPPower(Construction):
    def __init__(self,f,maxInput=10):
        self.f=f
        self.maxInput =maxInput

    def apply(self,G):
        return pptester.pppower(G,self.f)

    def isApplicable(self,G):
        if self.maxInput is not None and len(G.nodes)>self.maxInput:
            return False
        for v in self.f[0] + self.f[1]:
            if v.isnumeric() and int(v) not in G.nodes:
                return False
        return True

    def __str__(self):
        return str(self.f)


class ConstructionSimplePPDef(Construction):
    def __init__(self,G,x,y):
        self.G=G
        self.x=x
        self.y=y

    def isApplicable(self,G):
        return True

    def apply(self,G,ACWorks=False):
        return pptester.ppdefWithGraph(G,self.G,self.x,self.y,ACWorks)

    def __str__(self):
        return str((self.G.edges,self.x,self.y))


#gad = nx.DiGraph([(2,1)])
#gad.add_node(0)
#con = Poset.ConstructionGeneralPPPower(gad,(1,0),(0,2))
#G=con.apply(pptester.getDiCycle(2))

#TODO add constants
class ConstructionGeneralPPPower(Construction):
    def __init__(self,G,x,y,f=None,maxInput = None):
        self.G=G
        self.x=x
        self.y=y
        self.f=f
        self.maxInput = maxInput

        #add isolated nodes if necessary
        for v in x+y:
            if v not in G.nodes:
                G.add_node(v)
        if f is not None:
            for v in f:
                if v not in G.nodes:
                    G.add_node(v)

    def draw(self):
        pos={self.x[i]:(i,0) for i in range(len(self.x))}
        pos.update({self.y[i]:(i,2) for i in range(len(self.y))})
        pos.update({v:(i,1) for i,v in enumerate([u for u in self.G.nodes if not u in pos])})
        G = self.G.copy()
        relabel = {v: v for v in G.nodes}
        relabel.update({v:str(v)+':'+str(self.f[v]) for v in self.f})
        pos.update({str(v)+':'+str(self.f[v]):pos[v] for v in self.f})
        G = nx.relabel_nodes(G, relabel)

        nx.draw(G,pos,with_labels = True)

    def isApplicable(self,G):
        if self.maxInput is not None and len(G.nodes)>self.maxInput:
            return False
        if self.f is None:
            return True
        for v in self.f:
            if not self.f[v] in G.nodes:
                return False
        return True

    def apply(self,G,ACWorks=False,timelimit=float('inf')):
        #if self.f is None:
        #    return pptester.pppowerWithGraph(G,self.G,self.x,self.y,ACWorks,None,timelimit=timelimit)
        if self.f is None:
            fn = None
        else:
            fn = dict()
            for v in self.f:
                fn[v] = {self.f[v]}
        return pptester.pppowerWithGraph(G, self.G, self.x, self.y, ACWorks, fn,timelimit=timelimit)

    def __str__(self):
        return str((self.G.edges,self.x,self.y,self.f))

    def __eq__(self, other):
        if self.__str__() == other.__str__():#should not be necessary
            return True
        if not self.f is None == other.f is None:
            return False
        var = self.x+self.y
        varOth = other.x+other.y
        n = len(var)
        if n != len(varOth):
            return False
        #check equalities
        for i in range(n):
            for j in range(i+1,n):
                if (var[i] == var[j]) != (varOth[i] == varOth[j]):
                    return False


        f = {var[i]:{varOth[i]} for i in range(n)}
        g = {varOth[i]:{var[i]} for i in range(n)}
        #add constraints from constants
        if not self.f is None:
            for v in self.f:
                if v not in f:
                    f[v] = {u for u in other.f if other.f[u] == self.f[v]}
                else:
                    vOther = list(f[v])[0]
                    if vOther not in other.f or not other.f[vOther] == self.f[v]:
                        return False
            for v in other.f:
                if v not in f:
                    g[v] = {u for u in self.f if self.f[u] == other.f[v]}
                else:
                    vOther = list(g[v])[0]
                    if vOther not in self.f or not self.f[vOther] == other.f[v]:
                        return False

        if ArcCon.isHomEqACworks(self.G,other.G,g,f):
            if ArcCon.isHomEq(self.G,other.G,g,f):
                return True
        return False

#How to Use
#c=Poset.ConstructionGeneralPPPower(Structures.Structure([pptester.getPath('1'),pptester.getPath('1'),nx.DiGraph()]),(0,),(1,))
#c2=Poset.ConstructionGeneralPPPower(Structures.Structure([nx.DiGraph(),pptester.getPath('1'),nx.DiGraph()]),(0,),(1,))
#d=Poset.PPPowerBinarySignature([c1,c])
#P.applyConstruction(d)
#d=Poset.PPPowerBinarySignature([c1,c,c2])
#P.applyConstruction(d)
class PPPowerBinarySignature(Construction):
    def __init__(self,cs,maxInput = None):
        self.cs=cs
        for c in cs:
            if len(self.cs[0].G.Gs) != len(c.G.Gs):
                print('all constructions must use the same signature', [len(d.G.Gs) for d in cs])

    def isApplicable(self,G):
        return len(self.cs[0].G.Gs)==len(G.Gs)

    def apply(self,G,ACWorks=False,debug = False,timelimit=float('inf')):
        if debug:
            Gs = []
            for i,c in enumerate(self.cs):
                try:
                    Gs += [c.apply(G,ACWorks=ACWorks,timelimit=timelimit)]
                except:
                    raise Exception(i)
                print(len(Gs[-1].edges()))
            return Structures.Structure(Gs)
        return Structures.Structure([c.apply(G,ACWorks=ACWorks,timelimit=timelimit) for c in self.cs])

    def toTikz(self,colors=['green','red','dashed']):
        lines = []
        for j,c in enumerate(self.cs):
            lines += [r'\begin{tikzpicture}']
            for i,x in enumerate(c.x):
                lines += [r'\node[var-f,label=above:$x_{'+str(x)+'}$] ('+str(x)+') at ('+str(i)+',1) {};']
            for i,y in enumerate(c.y):
                lines += [r'\node[var-f,label=below:$y_{'+str(y)+'}$] ('+str(y)+') at ('+str(i)+',0) {};']
            lines += ['\\path[>=stealth\']']
            for color,G in enumerate(c.G.Gs):
                for e in G.edges():
                    loop = ',loop' if e[0]==e[1] else ''
                    lines += [r'('+str(e[0])+') edge[->,'+colors[color]+loop+'] ('+str(e[1])+')']
            lines += [r';']
            lines += [r'\node at (0.5,-1) {$\Phi_{'+str(j)+'}$};']
            lines += [r'\end{tikzpicture}']
        for l in lines:
            print(l)


def constructionFromString(t='(\'A\',\'a\')'):
    t=t.replace(' ','')
    # formula
    if t[1] == '\'':
        t = t.replace('\'', '')
        t = t.replace('(', '')
        t = t.replace(')', '')
        f= tuple(t.split(','))
        return ConstructionSimplePPPower(f)
    # general construction
    if t[:4] == '(Out':
        tup = ast.literal_eval(t.replace('OutEdgeView',''))+(None,)
        return ConstructionGeneralPPPower(nx.DiGraph(tup[0]),tup[1],tup[2],tup[3])
    return None
#    def __hash__(self):
#        return (self.G,self.x,self.y).__hash__()

gadgetadd111=nx.DiGraph([(0,2)])
gadgetadd111.add_nodes_from([1,4])
add111 = ConstructionGeneralPPPower(gadgetadd111,(0,1,2),(1,4,4)) #also makes N23 from T3, works for every graph without directed cycle?
#sub111 = ConstructionGeneralPPPower(pptester.getPath('110'),(0,),(3,))
T3FromTopDown4Constructsn4123 = ConstructionGeneralPPPower(nx.DiGraph([(0, 6),(1,7),(3,5)]), (0, 1, 2, 3,4), (5,6, 7, 5, 2),{4:3})
OrdLeqT3M13 = ConstructionGeneralPPPower(nx.DiGraph([(1,7), (2,8), (8,4),(2,7)]), (1,2,4,3),(6,7,7,8),{3:0,6:1})

def getUsualConstructions():
    cons = [
 ConstructionGeneralPPPower(nx.DiGraph(), (0,), (0,)), # construct C1
 ConstructionGeneralPPPower(nx.DiGraph([(1, 0)]), (0,), (1,)), # construct dual graph
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2)]), (0,), (1,)), #Pn+1 < Pn
 ConstructionGeneralPPPower(pptester.getPath('101'), (0,), (3,)),
 ConstructionGeneralPPPower(pptester.getPath('111'), (1,), (2,)), #get smooth subgraph
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (2, 3), (2, 1), (3, 4)]), (0,), (1,)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (3, 4), (3, 2)]), (3,), (4,)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 2), (3, 2), (3, 4),(4, 5)]), (0,), (1,)),
 ConstructionGeneralPPPower(nx.DiGraph([(2, 3), (2, 1), (4, 5), (4, 3), (1, 0)]), (0,), (2,)),
 ConstructionGeneralPPPower(nx.DiGraph([(3, 4), (3, 2), (4, 5), (1, 0), (2, 1)]), (4,), (5,)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (2, 1)]), (0, 1), (0, 2)),
 ConstructionGeneralPPPower(nx.DiGraph([(1, 2)]), (0, 1), (2, 0)), #C2 < C4
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 2), (0, 3), (1, 2), (4, 2), (4, 3)]), (1, 3), (4, 4)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (2, 0), (4, 0),(4,5),(5,6),(6,4)]), (4,), (0,)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (4, 3)]), (3, 0), (4, 1)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (0, 4), (1, 2), (1, 3)]), (4, 1), (2, 4)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (1, 2), (1, 3), (1, 4)]), (3, 4), (1, 1)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (1, 2), (1, 3), (4, 1)]), (2, 4), (1, 0)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (1, 2), (1, 3), (4, 2)]), (3, 4), (2, 2)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (1, 2), (1, 4), (3, 2)]), (0, 1), (3, 4)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (0, 3), (1, 2), (4, 3)]), (1, 1), (3, 2)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (1, 3), (4, 2)]), (1, 4), (3, 0)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (3, 0)]), (2, 0), (1, 3)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1)]), (0, 1), (2, 0)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 4), (1, 3)]), (0, 1, 2), (3 ,4, 1)),
 ConstructionGeneralPPPower(pptester.getPath('111'), (2, 0), (3, 1)), # T4 < M5
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (3, 1), (3, 2), (4, 2)]), (0, 3), (1, 4)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (1, 2), (3, 2), (3, 4), (5, 4)]), (0,), (5,)),
 ConstructionGeneralPPPower(nx.DiGraph([(2, 1)]), (0,1), (2,3),{0:1,3:4}),
 ConstructionGeneralPPPower(nx.DiGraph([(2, 1)]), (0, 2), (1, 3), {0: 4, 3: 0}),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 4)]), (0, 0, 2), (2, 4, 5)),
 ConstructionGeneralPPPower(nx.DiGraph([(4, 3), (6, 3), (6, 7), (7, 1)]), (6, 0, 7), (7, 4, 2), {0: 1}), # C2-1 < C1110
 ConstructionGeneralPPPower(nx.DiGraph([(0, 5),(2,4)]), (0, 1, 2, 3), (4, 5, 4, 1),{3:3}),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 5), (2, 4)]), (0, 1, 2, 3), (4, 5, 4, 1), {3: 0}),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 5), (2, 4)]), (0, 1, 2, 3), (4, 5, 4, 1), {3: 2}),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 4)]), (0, 0, 2), (2, 4, 5)),
 ConstructionGeneralPPPower(nx.DiGraph([(0, 2), (1, 5)]), (0,1,2), (3,2,5),{3:0}),
 ConstructionGeneralPPPower(nx.DiGraph([(4, 0)]), (0, 1, 2, 3), (4, 5, 1, 2), {3: 2,5:0}),#M5 < 21112
 ConstructionGeneralPPPower(nx.DiGraph([(0, 6), (6, 4), (1, 5), (7, 3), (3, 8), (8, 9), (9, 10), (2, 11), (12, 13), (13, 14), (14, 2)]), (0, 1, 2), (3, 4, 5))
 #ConstructionGeneralPPPower(nx.DiGraph([(0, 8), (9, 8), (9, 10), (10, 11), (12, 11), (12, 5), (2, 13), (14, 13), (14, 15), (15, 16), (17, 16), (17, 7), (18, 3), (3, 19), (19, 1), (1, 20)]), (0, 4, 2, 3), (1, 5, 4, 7)),
    ]
    return cons

def getLargeConstructions():
    return [ConstructionGeneralPPPower(nx.DiGraph([(12,17),(18,17),(18,2),(12,19),(20,19),(20,15),(1,21),(22,21),(22,12),(7,23),(24,23),(24,14)]),(1,2,7,4,1,7,7),(9,9,11,12,14,15,5),{4:1,11:2}),#T4M03 < T4M02
            ConstructionGeneralPPPower(nx.DiGraph([(0, 1), (20, 4), (4, 21), (22, 5), (5, 23), (24, 6), (6, 25), (26, 7), (7, 27)]),(0, 2, 4, 5, 6, 7), (1, 4, 5, 6, 7, 3), {2: 5, 3: 2}) #C31uT3 < n21312 u n313
            ]

def filterConstructionsForACWorks(G,Hs,power=3,gadgetSize=7,numberOfConstants=2,constantsFrom=[0,1],existential = False):
    while True:
        if existential:
            c = getSingleRandomConstruction(power, gadgetSize, numberOfConstants=numberOfConstants, constantsFrom=constantsFrom)
        else:
            c = getSingleRandomConstructionNoExistantial(power,gadgetSize,numberOfConstants=numberOfConstants,constantsFrom=constantsFrom)
        PG=c.apply(G)
        for H in Hs:
            if ArcCon.isHomEqACworks(PG,H):
                print(len(H.nodes),c)
                break


def getConstructions(power,maxGadgetSize,maxEdges=100,numberOfConstants=0,constantsFrom=[]):
    Gs = set()
    for i in range(2,maxGadgetSize+1):
        #todo also dissconnected gadgets are fine
        Gs = Gs.union(pptester.getReallyAllGraphs(i,False,maxEdges))
    Gads = []
    #print([(G.nodes,G.edges) for G in Gs])
    for G in Gs:
        for x in pptester.getTuples(power, G.nodes):
            for y in pptester.getTuples(power, G.nodes):
                for domain in pptester.getTuples(numberOfConstants,G.nodes): #TODO no repetations
                    for constants in pptester.getTuples(numberOfConstants,constantsFrom):
                        f = dict()
                        for i in range(numberOfConstants):
                            f[domain[i]]=constants[i]
                        nGad = ConstructionGeneralPPPower(G,x,y,f)
                        new = True
                        if x == y:
                            new = False
                        else:
                            for Gad in Gads:
                                if Gad == nGad:
                                    new = False
                                    break
                        if new:
                            Gads += [nGad]
    return Gads

def getSingleRandomConstruction(power,gadgetSize,edgepropability=0.1, minEdges = 1, maxEdges=100,numberOfConstants=0,constantsFrom=[],maxInput = None):
    G = pptester.getRandomGraph(gadgetSize,edgepropability, minEdges, maxEdges)
    x = random.choices(list(G.nodes), k=power)
    y = random.choices(list(G.nodes), k=power)
    domain = random.choices( list(G.nodes),k=numberOfConstants)
    constants = random.choices(list(constantsFrom),k=numberOfConstants)
    f = dict()
    for i in range(numberOfConstants):
        f[domain[i]]=constants[i]
    return ConstructionGeneralPPPower(G,x,y,f,maxInput)


def getSingleRandomConstructionNoExistantial(power,gadgetSize,edgepropability=0.1, minEdges = 1, maxEdges=100,numberOfConstants=0,constantsFrom=[]):
    if gadgetSize>2*power:
        gadgetSize=2*power
        print('gadgetSize can be at most: ', 2*power)
    G = pptester.getRandomGraph(gadgetSize,edgepropability, minEdges, maxEdges)
    vars = list(G.nodes)+random.choices(list(G.nodes), k=2*power-len(G.nodes))
    random.shuffle(vars)
    x = vars[:power]
    y = vars[power:]
    domain = random.choices(list(G.nodes),k=numberOfConstants)
    constants = random.choices(list(constantsFrom),k=numberOfConstants)
    f = dict()
    for i in range(numberOfConstants):
        f[domain[i]]=constants[i]
    return ConstructionGeneralPPPower(G,x,y,f)


class PosetData:
    def __init__(self,P = None):
        if P == None:
            self.Graph = nx.DiGraph()
            self.GraphWhole = nx.DiGraph()
            self.Graphs = dict() # GraphId -> Graph
            self.Names = dict() # GraphId -> Name
            self.classes = dict() # ClassId -> [GraphIds]
            self.edges = dict() # (GraphId,GraphId) -> phi or comment
            self.SeparationGraph = nx.DiGraph()
            self.SeparationGraphWhole = nx.DiGraph()
            self.SeparationEdges = dict() # (gid,gid) -> comment
            self.Ids = dict()  # ClassId -> [{Sat. Ids},{Unsat. Ids}]
            self.ACWorks=False
        else:
            self.Graph = P.Graph
            self.GraphWhole = P.GraphWhole
            self.Graphs = P.Graphs # GraphId -> Graph
            self.Names = P.Names # GraphId -> Name
            self.classes = P.classes # ClassId -> [GraphIds]
            self.edges = P.edges # (GraphId,GraphId) -> phi
            self.SeparationGraph = P.SeparationGraph
            self.SeparationGraphWhole = P.SeparationGraphWhole
            self.SeparationEdges = P.SeparationEdges
            self.Ids = P.Ids  # ClassId -> [{Sat. Ids},{Unsat. Ids}]
            self.ACWorks = P.ACWorks

#return pp-power for A<A' <-> B and h1:A' -> B, h2:B -> A'
#this d´function does not add existential variables
#deal with = by adding [(v,v) for v in A] as a relation to A
#A=G, B=H
#h2 o h1 should be the identity
def getConstructionFromHomomorphisms(G,H,h1=None,h2=None):
    if h1 is not None:
        print('h1 o h2 should be the identity, if yes then h1 is not necessary, if no then the algorithm possibly does not work')
    cs = []
    n = len(h2[list(H.nodes)[0]])

    for HE in H.Gs:
        Gsc = []
        x = tuple(range(n))
        y = tuple(range(n, 2 * n))

        for GE in G.Gs:
            Gc = nx.DiGraph()
            Gc.add_nodes_from(range(2*n))
            for u in Gc.nodes:
                for v in Gc.nodes:
                    #if forall (a,b) in HE (h2(a),h2(b))[u,v] in GE then add GE edge (u,v)
                    addEdge = True
                    for (a,b) in HE.edges:
                        u2,v2 = (h2[a]+h2[b])[u], (h2[a]+h2[b])[v]
                        if not GE.has_edge(u2,v2):
                            addEdge = False
                            break
                    if addEdge:# and ((u<n and v<n) or (u>=n and v>=n)):
                        Gc.add_edge(u,v)
            Gsc +=[Gc]
        c = ConstructionGeneralPPPower(Structures.Structure(Gsc),x,y)
        cs += [c]
    return PPPowerBinarySignature(cs)



class Poset:
    def __init__(self,P = None):
        if P == None:
            self.Graph = nx.DiGraph()
            self.GraphWhole = nx.DiGraph()
            self.Graphs = dict() # GraphId -> Graph
            self.Names = dict() # GraphId -> Name
            self.classes = dict() # ClassId -> [GraphIds]
            self.edges = dict() # (GraphId,GraphId) -> phi or comment
            self.SeparationGraph = nx.DiGraph()
            self.SeparationGraphWhole = nx.DiGraph()
            self.SeparationEdges = dict() # (gid,gid) -> comment
            self.Ids = dict() # ClassId -> [{Sat. Ids},{Unsat. Ids}]
            self.ACWorks = False
        else:
            self.Graph = P.Graph
            self.GraphWhole = P.GraphWhole
            self.Graphs = P.Graphs # GraphId -> Graph
            self.Names = P.Names # GraphId -> Name
            self.classes = P.classes # ClassId -> [GraphIds]
            self.edges = P.edges # (GraphId,GraphId) -> phi
            self.SeparationGraph = P.SeparationGraph
            self.SeparationGraphWhole = P.SeparationGraphWhole
            self.SeparationEdges = P.SeparationEdges
            self.Ids = P.Ids  # ClassId -> [{Sat. Ids},{Unsat. Ids}]
            try:
                self.ACWorks = P.ACWorks
            except:
                self.ACWorks = False

    def applyUsualIds(self):
        self.applyIdentities([Identities.Const,Identities.Malt,Identities.HM2,Identities.HM3,Identities.HM4,Identities.HM5,Identities.Sigma2,Identities.Sigma3,Identities.GuardedFS2,Identities.GuardedFS3,Identities.NU3])

    def addUsualGraphs(self,addOnlyTS=True,findNPhard=True):

        T3id = self.addGraph(pptester.T3,'T3')
        T4id = self.addGraph(pptester.T4,'T4')
        T5id = self.addGraph(pptester.T5,'T5')
        self.addGraph(pptester.T6, 'T6')
        self.addGraph(pptester.OrdDi,'Ord')
        C1id = self.addGraph(nx.DiGraph([(0,0)]),'C1')
        if not addOnlyTS:
            C2id = self.addGraph(nx.DiGraph([(0,1),(1,0)]),'C2')
            C3id = self.addGraph(nx.DiGraph([(0,1),(1,2),(2,0)]),'C3')
            self.addGraph(nx.DiGraph([(0,1),(1,2),(2,3),(3,0)]),'C4')
            C5id = self.addGraph(pptester.C5,'C5')
            self.Ids[self.getCId(C5id)][1].add(Identities.Sigma5)

        P2id = self.addGraph(nx.DiGraph([(0,1)]),'P2')
        self.addGraph(nx.DiGraph([(0,1),(1,2)]),'P3')
        self.addGraph(nx.DiGraph([(0,1),(1,2),(2,3)]),'P4')
        self.addGraph(nx.DiGraph([(0,1),(1,2),(2,3),(3,4)]),'P5')

        self.addGraph(pptester.M5,'M5')
        G = pptester.getCycle('11100')
        self.addGraph(pptester.getCycle('11100'),'N32')


        self.applyFormulas(pptester.getUsualFormulas()[:2],ACWorks=self.ACWorks)
        self.contract()

        #add separation edges
        self.applyIdentities([Identities.Const,Identities.Malt,Identities.Majority,Identities.HM2,Identities.HM3,Identities.HM4],ACWorks=self.ACWorks)
        self.findSeparationsFromIds()

        # apply usual formulas and constructions
        self.applyFormulas(pptester.getUsualFormulas(),ACWorks=self.ACWorks)
        self.applyConstructions(getUsualConstructions(),ACWorks=self.ACWorks)
        #ls = [self.applyFormula(f) for f in pptester.getUsualFormulas()]
        if findNPhard:
            for Gid in self.Graphs:
                if pptester.isSmooth(self.Graphs[Gid]) and not pptester.isUnionOfCycles(self.Graphs[Gid]):
                    for Hid in self.Graphs:
                        self.addEdge(Gid,Hid,'smooth')
        self.contract()

    def addHMImplications(self,n):
        for cid in self.classes:
            if Identities.getHM(2) in self.Ids[cid][1]:
                self.addIdentity(cid, False, Identities.Malt)
            for i in range(2,n):
                if Identities.getHM(i) in self.Ids[cid][0]:
                    self.addIdentity(cid,True,Identities.getHM((i+1)))
                if Identities.getHM(n-i+2) in self.Ids[cid][1]:
                    self.addIdentity(cid, False, Identities.getHM((n-i+1)))

    def addBlockerEdges(self):
        for Cid in self.classes:
            T3id = self.getIdByName('T3')
            if Identities.Malt in self.Ids[Cid][1] and T3id is not None:
                self.addEdge(self.classes[Cid][0],T3id,'HM1')

            for i in range(2,8):
                Tnid = self.getIdByName('T'+str(i+2))
                if Identities.getHM(i) in self.Ids[Cid][1] and Tnid is not None:
                    self.addEdge(self.classes[Cid][0], Tnid, 'HM'+str(i))


            C2id = self.getIdByName('C2')
            if Identities.Sigma2 in self.Ids[Cid][1] and C2id is not None:
                self.addEdge(self.classes[Cid][0],C2id,'Sigma2')


            C3id = self.getIdByName('C3')
            if Identities.Sigma3 in self.Ids[Cid][1] and C3id is not None:
                self.addEdge(self.classes[Cid][0],C3id,'Sigma3')


    def findNPHard(self,startId=-1,onlyOne=False,timelimit=float('inf'),sigg=True, onlyId=False):
        hardClassId=None
        if not onlyId:
            print('smoothness test (Borto Kozik)')

            ks =list(self.classes.keys())
            for Cid in ks:
                if Cid > startId and Cid in self.classes:
                    if len(self.Ids[Cid][0]) == 0:
                        print(Cid)
                        for Gid in self.classes[Cid]:
                            if pptester.isSmooth(self.Graphs[Gid]) and not pptester.isUnionOfCycles(self.Graphs[Gid]):
                                if hardClassId is None:
                                    hardClassId=Cid
                                    for Cid2 in self.classes:
                                        self.addEdge(Gid, self.classes[Cid2][0], 'smooth')
                                else:
                                    self.addEdge(Gid, self.classes[hardClassId][0], 'smooth')
                                self.contract()
                                if onlyOne:
                                    return
                                break
            self.contract()

            ks =list(self.classes.keys())
            print('smooth core test')
            for Cid in ks:
                if Cid > startId and Cid in self.classes:
                    if len(self.Ids[Cid][0]) == 0:
                        print(Cid)
                        for Gid in self.classes[Cid]:
                            G=self.Graphs[Gid].copy()
                            #get smooth subgraph
                            notSmooth = [v for v in G.nodes if G.in_degree[v]==0 or G.out_degree[v]==0]
                            while len(notSmooth)>0:
                                G.remove_nodes_from(notSmooth)
                                notSmooth = [v for v in G.nodes if G.in_degree[v] == 0 or G.out_degree[v] == 0]
                            #get core
                            G=pptester.getCore(G)
                            if pptester.isSmooth(G) and not pptester.isUnionOfCycles(G):
                                if hardClassId is None:
                                    hardClassId = Cid
                                    for Cid2 in self.classes:
                                        self.addEdge(Gid,self.classes[Cid2][0], 'smooth core')
                                else:
                                    self.addEdge(Gid, self.classes[hardClassId][0], 'smooth core')
                                self.contract()
                                if onlyOne:
                                    return
                                break
            self.contract()
        ks =list(self.classes.keys())
        if sigg:
            print('sigger tests')
            for Cid in ks:
                if Cid > startId and Cid in self.classes:
                    if len(self.Ids[Cid][0]) == 0:
                        print(Cid)
                        Gid = self.classes[Cid][0]
                        try:
                            if Identities.Sigg3 in self.Ids[Cid][1] or not Identities.satisfysIdentity(self.Graphs[Gid],Identities.Sigg3,timelimit=timelimit):
                                if hardClassId is None:
                                    hardClassId = Cid
                                    for Cid2 in self.classes:
                                        self.addEdge(Gid, self.classes[Cid2][0], 'siggers')
                                else:
                                    self.addEdge(Gid, self.classes[hardClassId][0], 'siggers')
                                print('contract')
                                self.contract()
                                if onlyOne:
                                    return

                        except:
                            print('timeout, Cid:', Cid,Gid)
                            if timelimit==float('inf'):
                                return

            self.contract()

    def getIdByName(self,name):
        ids = [gid for gid in self.Names.keys() if self.Names[gid] == name]
        if len(ids)==0:
            return None
        return ids[0]

    def printConstructions(self,source=None,target=None):
        if self.getIdByName(source) is not None:
            source = self.getIdByName(source)
        if self.getIdByName(target) is not None:
            target = self.getIdByName(target)

        for e in self.edges:
            if (source is None or source == e[0]) and (target is None or target == e[1]):
                print(self.getNameOrId(e[0]),self.getNameOrId(e[1]), self.edges[e])


    def getConstructionsUsed(self,maxInput=None):
        res = []
        for e in self.edges:
            res += [str(self.edges[e])]
        res = set(res)
        #print(res)
        cs = [constructionFromString(r) for r in res if constructionFromString(r) is not None]
        for c in cs:
            c.maxInput = maxInput
        return cs

    #return cid for graph given by gid
    def getCId(self,gid):
        cids = [cid for cid in self.classes.keys() if gid in self.classes[cid]]
        if len(cids)==0:
            return None
        return cids[0]

    def drawGraph(self,gid):
        name =''
        if gid in self.Names:
            name = ' name:'+self.Names[gid]
        pptester.drawGraph(self.Graphs[gid],"class:"+str(self.getCId(gid))+" graph:"+str(gid)+name)



    # add G up to homACEquiv
    def addGraph(self,G,Name = None,maxSize=100000,ACWorks=False, relabel = False,timelimit =2):
        #print('try to add (non core) Graph with ',len(G.nodes),'vertices')
        if len(G.nodes) == 0:
            return 'no empty graph'
        for id in self.classes.keys():
            for Hid in self.classes[id]:
                #TODO make keboard interrupt
                try:
                    if ArcCon.isHomEqACworks(G,self.Graphs[Hid]) and (self.ACWorks or ArcCon.isHomEq(G,self.Graphs[Hid],timelimit=timelimit)):
                        if Name != None:
                            self.Names[Hid] = Name
                        return Hid
                except:
                    print('timeout while adding graph', len(G.nodes),self.getNameOrId(Hid))
                    return None

        return self.addNewGraph(G,Name,maxSize,relabel=relabel)

    def computeHomorder(self):
        for gid1 in self.Graphs:
            for gid2 in self.Graphs:
                if gid1!=gid2:
                    if ArcCon.existsHom(self.Graphs[gid1],None,self.Graphs[gid2]):
                        self.addEdge(gid1,gid2,'hom')

    def addNewGraph(self,G,Name = None,maxSize=100000,relabel=False):
        try:
            GC = pptester.getCore(G,timelimit=10)
        except:
            print('timeout while computing core of new graph, nodes:',len(G.nodes),'edges:',len(G.edges))
            return 'timeout'
        if len(GC.nodes) > maxSize:
            return 'graph too large'
        if relabel:
            GC = pptester.relabelGraph(GC)
        gid = 0
        while gid in self.Graphs.keys():
            gid += 1
        graphId = gid
        cid = 0
        while cid in self.classes.keys():
            cid += 1
        classId = cid
        self.classes[classId] = [graphId]
        self.Graphs[graphId] = GC
        self.Graph.add_node(classId)
        self.GraphWhole.add_node(graphId)
        self.SeparationGraph.add_node(classId)
        self.SeparationGraphWhole.add_node(graphId)
        self.Ids[classId] = [set(), set()]
        if Name != None:
            self.Names[graphId] = Name
        print('\033[92mgraph added:\033[0m ',self.getNameOrId(graphId))
        return graphId


    def isThereAnEdgeContradiction(self):
        return False
        # todo

    def addTSAllNId(self):
        for cid in self.classes:
            if TSAllN not in self.Ids[cid][0] and TSAllN not in self.Ids[cid][1]:
                self.addIdentity(cid,pptester.isTotallySymmetric(self.Graphs[self.classes[cid][0]]),TSAllN)


    def contractClasses(self,id1,id2):
        #add edges
        for id3 in self.Graph.neighbors(id2):
            if id1 != id3 and not self.Graph.has_edge(id1, id3):
                self.Graph.add_edge(id1,id3)
                #self.edges[(id1,id3)] = self.edges[(id2,id3)]
        for id3 in self.Graph.predecessors(id2):
            if id1 != id3 and not self.Graph.has_edge(id3, id1):
                self.Graph.add_edge(id3,id1)
                #self.edges[(id3,id1)] = self.edges[(id3,id2)]

        #add edges in separation graph
        for id3 in self.SeparationGraph.neighbors(id2):
            if id1 != id3 and not self.SeparationGraph.has_edge(id1, id3):
                self.SeparationGraph.add_edge(id1, id3)
        for id3 in self.SeparationGraph.predecessors(id2):
            if id1 != id3 and not self.SeparationGraph.has_edge(id3, id1):
                self.SeparationGraph.add_edge(id3, id1)

        #merge identities
        self.Ids[id1][0] = self.Ids[id1][0].union(self.Ids[id2][0])
        self.Ids[id1][1] = self.Ids[id1][1].union(self.Ids[id2][1])

        #remove vertex
        self.Graph.remove_node(id2)
        self.SeparationGraph.remove_node(id2)
        self.classes[id1] += self.classes.pop(id2)




    def addGraphs(self,Gs, onlyTS = False):
        #todo add ACWorks
        for i,G in enumerate(Gs):
            if (not onlyTS) or pptester.isTotallySymmetric(G):
                ret = self.addGraph(G)
                if ret is None:
                    print('graphindex',i)

    def removeNoTSGraphs(self):
        Gids = [Gid for Gid in self.Graphs if not pptester.isTotallySymmetric(self.Graphs[Gid])]
        for Gid in Gids:
            print('remove',self.getNameOrId(Gid))
            self.removeGraph(Gid)


    #use: Gs= [pptester.getPath(w) for w in pptester.getWords(6,'01')]
    def applyPathsPPDef(self,Ps,output= True,outputDebug = False, ACWorks = False, timelimit=1.4):
        for i, p in enumerate(Ps):
            if output:
                print(i, str(p.edges))
            for i in range(len(p.nodes)):
                for j in range(i+1,len(p.nodes)):
                    if ArcCon.isTreeCore(p,{i:{i},j:{j}}):
                        #print((p,i,j,ACWorks))
                        construction = ConstructionSimplePPDef(p,i,j)
                        self.applyConstruction(construction, output, outputDebug, self.ACWorks, timelimit)

    def applyPaths(self,length,power,addFreeNodes=0,timelimit = 1.4):
        n=power
        Ps = [pptester.getPath('0'+w) for w in pptester.getWords(length-1,'01')]
        for i,p in enumerate(Ps):
            p.add_nodes_from(range(n+1,n+1+addFreeNodes))
            xys = list(itertools.product(list(p.nodes),repeat = n))
            for i in range(len(xys)):
                for j in range(i+1,len(xys)):
                    x = xys[i]
                    y=xys[j]
                    if ArcCon.isTreeCore(p,{i:{i} for i in x+y}):
                        c = ConstructionGeneralPPPower(p,x,y)
                        print(i,c)
                        self.applyConstruction(c)

    def applyFormulas(self,fs,output= True,outputDebug = False, ACWorks = False, timelimit=1.4,addGraphs=False,maxSize=10,maxInput=10,applyTo=None):
        for i,f in enumerate(fs):
            if output:
                print(i,f)
            construction = ConstructionSimplePPPower(f,maxInput)
            self.applyConstruction(construction,output,outputDebug,self.ACWorks,timelimit,addGraphs,maxSize,applyTo=applyTo)
            #self.applyFormula(f,output,outputDebug,ACWorks,timelimit)

#P.applyConstructions([Poset.getSingleRandomConstruction(2,4,minEdges=1,numberOfConstants=1,constantsFrom=[0,1,2,3,4]) for _ in range(100)])
#P.applyConstructions([Poset.getSingleRandomConstruction(3,7,0.1,numberOfConstants=2,constantsFrom=range(4),maxInput=5) for _ in range(500)],addGraphs=True,maxSize=10,applyTo=P.getIdByName('T4'))
    def applyConstructions(self,cons,output= True,outputDebug = False, ACWorks = False, timelimit=1.4,addGraphs=False,maxSize=10,applyTo=None):
        for i,c in enumerate(cons):
            if output:
                print(i,c)
            self.applyConstruction(c,output,outputDebug,self.ACWorks,timelimit,addGraphs=addGraphs,maxSize=maxSize,applyTo=applyTo)


    # use f to separate classes
    def applyFormula(self, f, output=True, outputDebug=False, ACWorks=False, timelimit=1.4):
        for id1 in self.classes.keys():
            for Gid in self.classes[id1]:
                # test whether all constats used in f also occur in G
                applicable = True
                for v in str(f):
                    if v.isnumeric() and int(v) not in self.Graphs[Gid].nodes:
                        applicable = False
                        break
                if not applicable:
                    break
                if outputDebug:
                    print("class:", id1, "graph:", Gid)
                PG = None #PG = pptester.pppower(self.Graphs[Gid], f)

                #if outputDebug:
                #    print("class:", id1, "graph:", Gid, len(PG.nodes))
                #if len(PG.nodes) > 500:
                #    break

                # compute core?
                foundEdge = False
                for id2 in self.classes.keys():
                    if id1 != id2 and not self.Graph.has_edge(id1, id2) and not self.SeparationGraph.has_edge(id1, id2):
                        for Hid in self.classes[id2]:
                            if outputDebug:
                                print("class2:", id2, "graph:", Hid)
                            try:
                                if PG is None:
                                    PG=pptester.pppower(self.Graphs[Gid], f)
                                #if ArcCon.isHomEqACworks(PG, self.Graphs[Hid]) and (
                                #        self.ACWorks or ArcCon.isHomEq(PG, self.Graphs[Hid], timelimit=timelimit)):
                                if ArcCon.existsHom(PG, None, self.Graphs[Hid],
                                                    self.ACWorks or TSAllN in self.Ids[id2][0],
                                                    timelimit=timelimit) and ArcCon.existsHom(self.Graphs[Hid],
                                                                                              None, PG,
                                                                                              self.ACWorks or TSAllN in
                                                                                              self.Ids[id1][0],
                                                                                              componentwise=False,
                                                                                              timelimit=timelimit):

                                    # if isHomEqLimitedTime(PG,self.Graphs[Hid]):
                                    # if isHomEqInterruptable(PG,self.Graphs[Hid]):
                                    if not self.GraphWhole.has_edge(Gid, Hid):
                                        self.addEdge(Gid, Hid, str(f))
                                    foundEdge = True
                                    break
                            except Exception:
                                print('timeout',self.getNameOrId(Gid), self.getNameOrId(Hid), f)
                                pass
                    if foundEdge:
                        break

    def applyConstruction(self,construction:Construction,output= True,outputDebug = False, ACWorks = False, timelimit=1.4,addGraphs = False,maxSize=10,applyTo=None):
        for id1 in self.classes.keys():
            for Gid in self.classes[id1]:
                if applyTo is not None:
                    Gid = applyTo
                    id1 = self.getCId(Gid)
                # test whether all constants used in f also occur in G
                if not construction.isApplicable(self.Graphs[Gid]):
                    break
                if outputDebug:
                    Glabel = ''
                    if Gid in self.Names:
                        Glabel = self.Names[Gid]
                    print("class:",id1,"graph:",Gid,Glabel)
                PG = None #construction.apply(self.Graphs[Gid])


                if outputDebug:
                    print("class:",id1,"graph:",Gid,PG.edges if PG is not None else '')
                #if len(PG.nodes) > 500:
                #    break



                found = False

                #compute core?
                for id2 in self.classes.keys():
                    if not found and id1 != id2 and not self.Graph.has_edge(id1,id2) and not self.SeparationGraph.has_edge(id1,id2):
                        for Hid in self.classes[id2]:
                            if outputDebug:
                                print("class2:",id2,"graph:",Hid)
                            if PG is None:
                                PG = construction.apply(self.Graphs[Gid])

                                # reduce components of PG
                                #if ACWorks or TSAllN in self.Ids[id1][0]:
                                    #print(len(PG.nodes))
                                #    PG=ArcCon.reduceHomComponentsACWorks(PG)
                                    #print(len(PG.nodes))

                            try:
                                if ArcCon.existsHom(PG,None,self.Graphs[Hid],self.ACWorks or TSAllN in self.Ids[id2][0],timelimit=timelimit) and ArcCon.existsHom(self.Graphs[Hid],None,PG,self.ACWorks or TSAllN in self.Ids[id1][0],componentwise=False,timelimit=timelimit):
                                #if ArcCon.isHomEqACworks(PG,self.Graphs[Hid]) and (self.ACWorks or (TSAllN in self.Ids[id1][0] and TSAllN in self.Ids[id2][0]) or ArcCon.isHomEq(PG,self.Graphs[Hid],timelimit=timelimit)):
                                    #if isHomEqLimitedTime(PG,self.Graphs[Hid]):
                                    #if isHomEqInterruptable(PG,self.Graphs[Hid]):
                                    if not self.GraphWhole.has_edge(Gid,Hid):
                                        self.addEdge(Gid,Hid,construction)
                                    found = True
                                    break
                            except Exception as e:

                                print('timeout',self.getNameOrId(Gid), self.getNameOrId(Hid),str(self.Graphs[Gid]),e)
                                found = True
                                pass

                if addGraphs and not found and not PG is None:
                    Gids = list(self.Graphs.keys())
                    Hid = self.addGraph(PG,maxSize=maxSize,relabel=True)
                    #print(Hid,Gids)
                    if Hid in self.Graphs and not Hid in Gids:
                        self.addEdge(Gid, Hid, construction)
                        self.Ids[self.getCId(Hid)][0].update(copy.deepcopy(self.Ids[self.getCId(Gid)][0]))
                        return
                if applyTo is not None:
                    return





    def removeGraph(self,Gid):
        self.Graphs.pop(Gid)
        if Gid in self.Names:
            self.Names.pop(Gid)
        self.GraphWhole.remove_node(Gid)
        self.SeparationGraphWhole.remove_node(Gid)

        Cid = self.getCId(Gid)
        self.classes[Cid].remove(Gid)
        if len(self.classes[Cid]) == 0:
            self.classes.pop(Cid)
            self.Graph.remove_node(Cid)
            self.SeparationGraph.remove_node(Cid)

    def getNameOrId(self,Gid):
        if Gid in self.Names:
            return self.Names[Gid]
        return str(Gid)

    def addSeparationEdge(self,gid1,gid2,comment):
        #if self.SeparationGraphWhole.has_edge(gid1, gid2):
        if self.SeparationGraph.has_edge(self.getCId(gid1), self.getCId(gid2)):
            return False
        print('\033[92madd Separationedge\033[0m', self.getNameOrId(gid1), self.getNameOrId(gid2), comment)
        self.SeparationGraphWhole.add_edge(gid1, gid2)
        self.SeparationEdges[(gid1, gid2)] = comment
        self.addSeparationEdgePropagate(self.getCId(gid1), self.getCId(gid2))
        return True

    def addSeparationEdgePropagate(self,cid1,cid2):
        if cid1 == cid2:
            print('error in separation edges (class can not construct itself)',cid1,cid2)
            return
        if self.Graph.has_edge(cid1,cid2):
            print('error class can construct itself but also can not',cid1,cid2)
            return
        if self.SeparationGraph.has_edge(cid1,cid2):
            return
        self.SeparationGraph.add_edge(cid1,cid2)
        # separation edges
        # (A sep <- B -> pp C) => (A sep <- C)
        # (A pp -> B <- sep C) => (A sep <- C)
        for cid3 in self.Graph.predecessors(cid2):
            self.addSeparationEdgePropagate(cid1,cid3)
        for cid0 in self.Graph.successors(cid1):
            self.addSeparationEdgePropagate(cid0,cid2)


    def addEdge(self,gid1,gid2,comment):
        #todo also udate Ids
        if self.GraphWhole.has_edge(gid1, gid2):
            return
        print('\033[92madd Edge\033[0m', self.getNameOrId(gid1), self.getNameOrId(gid2), comment)
        self.GraphWhole.add_edge(gid1, gid2)
        self.edges[(gid1, gid2)] = comment
        self.addEdgePropagate(self.getCId(gid1), self.getCId(gid2))

    def addEdgePropagate(self, id1, id2):
        if self.SeparationGraph.has_edge(id1,id2):
            print('error class can construct itself but also can not',id1,id2)
            return
        #print(id1,id2)
        if id1 != id2 and not self.Graph.has_edge(id1, id2):
            #pp edges
            self.Graph.add_edge(id1, id2)
            for id3 in self.Graph.neighbors(id2):
                self.addEdgePropagate(id1,id3)
            for id0 in self.Graph.predecessors(id1):
                self.addEdgePropagate(id0,id2)

            #separation edges
            # (A sep <- B -> pp C) => (A sep <- C)
            # (A pp -> B <- sep C) => (A sep <- C)
            for id3 in self.SeparationGraph.predecessors(id2):
                self.addSeparationEdgePropagate(id3,id1)
            for id0 in self.SeparationGraph.successors(id1):
                self.addSeparationEdgePropagate(id2,id0)

    def eraseIds(self):
        for Cid in self.classes:
            self.Ids[Cid] = [set(),set()]
        #TODO self.SeparationGraph. remove all edges

    def contract(self):
        changed = True
        while changed:
            changed = False
            for id1 in self.Graph.nodes:

                for id2 in self.Graph.successors(id1):
                    if id1 != id2 and self.Graph.has_edge(id1,id2) and self.Graph.has_edge(id2,id1):
                        self.contractClasses(id1,id2)
                        print("contract",id1,id2)
                        changed = True
                        break
                        #return self.contract()
                if changed:
                    break

    def addIdentity(self,Cid,sat,Id,debug=True,propagate=True):
        if Id in self.Ids[Cid][0] or Id in self.Ids[Cid][1]:
            return
        if sat:
            if debug: print('add Identity', Cid, [self.getNameOrId(Gid) for Gid in self.classes[Cid] if Gid in self.Names], '\033[92m',sat,'\033[0m', Id)
            self.Ids[Cid][0].add(Id)
            if propagate:
                for Did in self.Graph.successors(Cid):
                    self.addIdentity(Did,sat,Id,debug)
        else:
            if debug: print('add Identity', Cid, [self.getNameOrId(Gid) for Gid in self.classes[Cid] if Gid in self.Names], '\033[91m',sat,'\033[0m', Id)
            self.Ids[Cid][1].add(Id)
            if propagate:
                for Did in self.Graph.predecessors(Cid):
                    self.addIdentity(Did,sat,Id,debug)

    def getIdBorders(self,Id):
        minimalSat = []
        maximalNonsat = []
        for gid in self.Graphs:
            if Id in self.Ids[self.getCId(gid)][0]:
                lowerCovers = self.Graph.predecessors(self.getCId(gid))
                if not (False in [Id in self.Ids[Cid][1] for Cid in lowerCovers]):
                    minimalSat+=[gid]

            if Id in self.Ids[self.getCId(gid)][1]:
                upperCovers = self.Graph.successors(self.getCId(gid))
                if not (False in [Id in self.Ids[Cid][0] for Cid in upperCovers]):
                    maximalNonsat += [gid]
        return (minimalSat,maximalNonsat)

    def applyIdentities(self,Ids,removeIdsThatDontGiveNewSeparations = True,timelimit=float('inf'),allBalanced = False,debug=True,maxTimeouts=float('inf'),printSatFraction=False,conservativeFirst=True,idempotent=True):
        GoodIds=[]
        if printSatFraction:
            usefullIds=0
            sat = 0
            unsat = 0
        for i,Id in enumerate(Ids):
            print(i,Id)
            if conservativeFirst:
                self.applyIdentity(Id, self.ACWorks, timelimit=0.5, allBalanced=allBalanced, debug=debug,
                                   maxTimeouts=maxTimeouts,idempotent=idempotent)
                self.applyIdentity(Id,self.ACWorks,timelimit=1,allBalanced=allBalanced,debug=debug,maxTimeouts=maxTimeouts,conservative=True,idempotent=True)
            self.applyIdentity(Id,self.ACWorks,timelimit=timelimit,allBalanced=allBalanced,debug=debug,maxTimeouts=maxTimeouts,idempotent=idempotent)
            if printSatFraction:
                satt = 100*len([cid for cid in self.classes if Id in self.Ids[cid][0]])/len(self.Graph.nodes)
                unsatt = 100*len([cid for cid in self.classes if Id in self.Ids[cid][1]])/len(self.Graph.nodes)
                if satt > 0 and unsatt > 0:
                    usefullIds += 1
                print('sat%,unsat%:',math.floor(satt),math.floor(unsatt))

                sat += satt
                unsat += unsatt
            if removeIdsThatDontGiveNewSeparations and not self.findSeparationsFromIds():
                #if no new separations then remove Id again
                for cid in self.classes:
                    if Id in self.Ids[cid][0]:
                        self.Ids[cid][0].remove(Id)
                    elif Id in self.Ids[cid][1]:
                        self.Ids[cid][1].remove(Id)
            else:
                GoodIds += [Id]
        if printSatFraction:
            print('overall sat%,unsat%:',math.floor(sat/len(Ids)),math.floor(unsat/len(Ids)),'usefull%',math.floor(100*usefullIds/len(Ids)))
        return GoodIds

    def renameIdentity(self,Id,newName):
        for cid in self.Ids:
            for Id2 in self.Ids[cid][0]:
                if Id2==Id:
                    Id2.Name=newName

            for Id2 in self.Ids[cid][1]:
                if Id2==Id:
                    Id2.Name=newName

    def runIdentity(self,Id:Identities.Identity,ACWorks = False,timelimit = float('inf'),allBalanced = False,conservative=False):
        for Cid in self.classes:
            try:
                if allBalanced:
                    sat = Identities.satisfysIdentity(self.Graphs[self.classes[Cid][0]], Id,
                                                          self.ACWorks or TSAllN in self.Ids[Cid][0],
                                                          timelimit=timelimit,partition=pptester.getLevelsOfBalancedGraph(self.Graphs[self.classes[Cid][0]]),conservative=conservative)
                else:
                    sat = Identities.satisfysIdentity(self.Graphs[self.classes[Cid][0]],Id,self.ACWorks or TSAllN in self.Ids[Cid][0],timelimit=timelimit,conservative=conservative)
                if sat:
                    if Id in self.Ids[Cid][1]:
                        print('\033[91mERROR\033[0m', Cid,Id)
                        return False
                    print('Identity', Cid, [self.getNameOrId(Gid) for Gid in self.classes[Cid] if Gid in self.Names],'\033[92m', sat, '\033[0m', Id)
                else:
                    if Id in self.Ids[Cid][0]:
                        print('\033[91mERROR\033[0m', Cid,Id)
                        return False
                    print('Identity', Cid, [self.getNameOrId(Gid) for Gid in self.classes[Cid] if Gid in self.Names],'\033[91m', sat, '\033[0m', Id)

            except:
                print('timeout:', self.getNameOrId(self.classes[Cid][0]),Id)
        return True


    def applyIdentity(self,Id:Identities.Identity,ACWorks = False,timelimit = float('inf'),allBalanced = False,debug=True,maxTimeouts=float('inf'),conservative=False,idempotent=True):
        if Id == TSAllN:
            return
        timeouts = 0
        nonTimeouts = 0
        for Cid in self.classes:
            if Id not in self.Ids[Cid][0] and Id not in self.Ids[Cid][1]:
                try:
                    if allBalanced:
                        sat = Identities.satisfysIdentity(self.Graphs[self.classes[Cid][0]], Id,
                                                          self.ACWorks or TSAllN in self.Ids[Cid][0],
                                                          timelimit=timelimit,partition=pptester.getLevelsOfBalancedGraph(self.Graphs[self.classes[Cid][0]]),conservative=conservative,Idempotent=idempotent)
                    else:
                        sat = Identities.satisfysIdentity(self.Graphs[self.classes[Cid][0]],Id,self.ACWorks or TSAllN in self.Ids[Cid][0],timelimit=timelimit,conservative=conservative,Idempotent=idempotent)
                    if not conservative or sat:
                        self.addIdentity(Cid,sat,Id,debug=debug)
                    nonTimeouts +=1
                except:
                    print('timeout:', self.getNameOrId(self.classes[Cid][0]),Id)
                    timeouts +=1
                    if timeouts > maxTimeouts:# or timeouts-nonTimeouts>0:
                        print('maxTimeouts reached',timeouts,nonTimeouts)
                        return


    def findSeparationsFromIds(self):
        new = False
        for Cid in self.classes:
            for Did in self.classes:
                if not self.SeparationGraph.has_edge(Cid,Did) and not self.Graph.has_edge(Cid,Did):
                    diff = self.Ids[Cid][0].intersection(self.Ids[Did][1])
                    if len(diff)>0:
                        sId = str(diff.pop())
                        if self.addSeparationEdge(self.classes[Cid][0],self.classes[Did][0],sId):
                            new = True
        return new

    def storetToFile(self,filename='poset.txt'):
        f = open(filename,'wb')
        pickle.dump(PosetData(self),f,pickle.HIGHEST_PROTOCOL)

    def drawOpenEdgesPoset(self):
        G = self.SeparationGraph.copy()

        G = nx.complement(G)
        G.remove_edges_from(self.Graph.edges)
        mapping = dict()
        for classId in self.classes.keys():
            label = str(classId) + ':'
            for GraphId in self.classes[classId]:
                if GraphId in self.Names.keys():
                    label += self.Names[GraphId] + ","
                else:
                    label += str(GraphId) + ','
            if label != '':
                mapping[classId] = label[:-1]
        G = nx.relabel_nodes(G, mapping)
        pptester.drawGraph(G, 'Unknown edges')


    def drawSeparationPoset(self):
        G = self.SeparationGraph.copy()
        mapping = dict()
        for classId in self.classes.keys():
            label = str(classId) + ':'
            for GraphId in self.classes[classId]:
                if GraphId in self.Names.keys():
                    label += self.Names[GraphId] + ","
                else:
                    label += str(GraphId) + ','
            if label != '':
                mapping[classId] = label[:-1]
        G = nx.relabel_nodes(G, mapping)
        pptester.drawGraph(G,'SeparationGraph')

    def drawSeparationPosetWhole(self):
        G = nx.relabel_nodes(self.SeparationGraphWhole, self.Names)
        pptester.drawGraph(G, 'SeparationGraphWhole')

    def getAllEdgesWith(self,nameOrId):
        id = self.getIdByName(nameOrId)
        if id is None:
            id = nameOrId
        return [(self.getNameOrId(e[0]),self.getNameOrId(e[1]),self.edges[e]) for e in self.edges if e[0]==id or e[1]==id]

    #[{SatIds},{UnsatIds},color]
    #P.colorDrawPoset([({Identities.HM2},set(),'red')])
    def colorDrawPoset(self,color=[]):
        mapping = dict()
        coloring = dict()
        for classId in self.classes.keys():
            label = str(classId) + ':'
            for GraphId in self.classes[classId]:
                if GraphId in self.Names.keys():
                    label += self.Names[GraphId] + ","
                else:
                    label += str(GraphId) + ','
            if label != '':
                mapping[classId] = label[:-1]
                #color
            for c in color:
                #print(c,self.Ids[classId])
                if c[0].issubset(self.Ids[classId][0]) and c[1].issubset(self.Ids[classId][1]):
                    coloring[mapping[classId]] = c[2]
        G = nx.relabel_nodes(self.Graph, mapping)
        #print(coloring)

        # remove transitive edges
        done = False
        edges = set(G.edges)
        while len(edges)>0:
            (u,v) = edges.pop()
            G.remove_edge(u, v)
            if not nx.has_path(G, u, v):
                #G.remove_edge(u, v)
                G.add_edge(u, v)

        pptester.colorDrawGraph(G, "PPPoset"+str(color),coloring)



    def drawPoset(self,withTransitive = False,withIdentitiesSat = True,withIdentitiesUnsat = False,returnEdges=False):
        mapping = dict()
        for classId in self.classes.keys():
            label = str(classId) + ':'
            for GraphId in self.classes[classId]:
                if GraphId in self.Names.keys():
                    label += self.Names[GraphId]+","
                else:
                    label += str(GraphId)+','
            if withIdentitiesSat:
                label += ' Sat:'
                for Id in self.Ids[classId][0]:
                    label += str(Id) + ','
            if withIdentitiesUnsat:
                label += ' Uns:'
                for Id in self.Ids[classId][1]:
                    label += str(Id) + ','

            if label != '':
                mapping[classId] = label[:-1]

        G = nx.relabel_nodes(self.Graph, mapping)
        if withTransitive:
            pptester.drawGraph(G,"PPPoset")
            return
        # remove transitive edges
        done = False
        edges = set(G.edges)
        while len(edges)>0:
            (u,v) = edges.pop()
            G.remove_edge(u, v)
            if not nx.has_path(G, u, v):
                #G.remove_edge(u, v)
                G.add_edge(u, v)

        pptester.drawGraph(G, "PPPoset (no transitive edges)")
        if returnEdges: return G.edges

def removeTransitiveEdges(G):
    edges = set(G.edges)
    while len(edges) > 0:
        (u, v) = edges.pop()
        G.remove_edge(u, v)
        if not nx.has_path(G, u, v):
            # G.remove_edge(u, v)
            G.add_edge(u, v)
    return G

def isHomEqInterruptable(G1,G2):
    try:
        return ArcCon.isHomEqACworks(G1, G2) and ArcCon.isHomEq(G1, G2)
    except:
        print('skip')
        pass
        return False
    return False


def isHomEqLimitedTime(G1,G2):
    manager = Manager()
    res = manager.dict()

    if run_with_limited_time(func=isHomEqLimitedTimeHelper,args=(G1,G2,res),timeout=10):
        return res[0]
    else:
        print('Timeout')
        return False



def isHomEqLimitedTimeHelper(G1,G2,result):
    if ArcCon.isHomEqACworks(G1, G2) and ArcCon.isHomEq(G1, G2):
        result[0] = True
    else:
        result[0] = False

#dont use, fucks up pycharm
def run_with_limited_time(func, args, kwargs=[], timeout=1):
    """Runs a function with time limit

    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    start = time.time()

    #executer = concurrent.futures.ThreadPoolExecutor()
    p = Process(target=func, args=args, kwargs=kwargs)

    #p = threading.Thread(target=func, args=args, kwargs=kwargs)

    p.start()
    print("start3")
    #i = 0
    #while p.is_alive() and i<timeout:
    #    print(i)
    #time.sleep(0.5)
    #    i += 0.05
    p.join(timeout=time) #fucking slow :(
    print(str("time " + str(time.time() - start)))
    print("start4")
    if p.is_alive():
        p.terminate()
        return False

    return True


def posetFromFile(filename='poset.txt'):
    #poset4.txt #4 element connected graphs without double edges
    #poset4Backup.txt #4 element connected graphs without double edges
    #poset4Double.txt # #4 element graphs
    #poset5TS.txt
    #poset5TSBackup.txt
    #posetLarge.txt
    #paths.txt
    #pathssmall.txt
    f = open(filename,'rb')

    try:
        return Poset(pickle.load(f))
    except AttributeError:
        print('something went wrong')
        f.close()
        f = open(filename, 'rb')
        return pickle.load(f)
#def signal_handler(sig, frame):
#    print('You pressed Ctrl+C!')
    #sys.exit(0)

#signal.signal(signal.SIGBREAK, signal_handler)
#print('Press Ctrl+C')
#time.sleep(10)

ColoringHM = [({Identities.Malt},set(),'red'),({Identities.HM2},{Identities.Malt},'yellow'),({Identities.HM3},{Identities.HM2},'green'),({Identities.HM4},{Identities.HM3},'orange'),({Identities.HM5},{Identities.HM4},'blue'),(set(),{Identities.HM5},'black')]
ColoringMaj = [({Identities.Majority},set(),'orange')]
ColoringGFS = [({Identities.GuardedFS2},set(),'red'),({Identities.GuardedFS3},{Identities.GuardedFS2},'yellow'),({Identities.GuardedFS4},{Identities.GuardedFS3},'green'),({Identities.GuardedFS5},{Identities.GuardedFS4},'orange'),(set(),{Identities.GuardedFS5},'blue')]


colors1 = ['#6ff6f6','#00aad0','#005f9b','blue']#blue
colors2 = ['#0f4103','#387922','#64b643','green']#green
colors3 = ['yellow','#fb6c1c','orange','red']#red

colors = [c for cs in [[colors1[i],colors2[i],colors3[i]] for i in range(len(colors1))] for c in cs]

def toHexColor(i):
    n = str(hex(i))
    return '#'+'0'*(6-len(n)+2)+n[2:]

def getColoringHM(n=10):
    res = [({Identities.Malt},set(),colors[0])]
    for i in range(1,n):
        res += [({Identities.getHM(i+1)},{Identities.getHM(i)},colors[i])]
    res += [(set(), {Identities.getHM(n)},colors[n])]
    return res

