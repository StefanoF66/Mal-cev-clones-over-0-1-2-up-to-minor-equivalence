import random

import networkx as nx

import ArcConFast
import Identities
import Structures
import Poset
import networkx as nx
import ast
import itertools

import pptester


def readClone(classNr):
    ls = open(r'3ElementsClones.txt').readlines()
    for i, l in enumerate(ls):
        if l.startswith('Class  ' + str(classNr)):
            line = i
    print(ls[line], 'in Line', line)
    Gs = []
    # add constants using loops
    urels = ast.literal_eval(ls[line + 1][15:])
    for rel in urels:
        Gs += [nx.DiGraph([(int(i), int(i)) for i in rel])]
    rels = ast.literal_eval(ls[line + 2][16:])
    for rel in rels:
        Gs += [nx.DiGraph([(int(e[0]), int(e[1])) for e in rel])]
    return Structures.Structure(Gs)


def readClones(debug=False):
    ls = open(r'3ElementsClones.txt').readlines()
    P = Poset.Poset()
    # print('number of lines:',len(ls))

    # read graphs
    for i in range(len(ls)):
        if ls[i].startswith('Minor Class'):
            GName = ls[i][13:-1]
            if debug: print(i, GName)
            Gs = []
            # add constants using loops
            urels = ast.literal_eval(ls[i + 6][15:])
            for rel in urels:
                Gs += [nx.DiGraph([(int(i), int(i)) for i in rel])]
            rels = ast.literal_eval(ls[i + 7][16:])
            for rel in rels:
                Gs += [nx.DiGraph([(int(e[0]), int(e[1])) for e in rel])]
            id = P.addGraph(Structures.Structure(Gs), GName)
            if debug: print(id)
    # return P
    # read edges
    for i in range(len(ls)):
        if ls[i].startswith('Minor Class'):
            GName = ls[i][13:-1]
            aboves = ast.literal_eval(ls[i + 1][7:])
            belows = ast.literal_eval(ls[i + 2][7:])
            if debug: print(GName, aboves, belows)
            for above in aboves:
                if debug: print(above)
                P.addEdge(P.getIdByName(GName), P.getIdByName(str(above)), 'see line ' + str(i + 1))
            for below in belows:
                if debug: print(below, P.getIdByName(str(below)))
                P.addEdge(P.getIdByName(str(below)), P.getIdByName(GName), 'see line ' + str(i + 2))
    return P


def readEdges():
    clonesLines = open(r'3ElementsClones.txt').readlines()
    edgesLines = open(r'JanEdges.txt').readlines()
    edges = []
    q = dict()
    for l in clonesLines:
        if l.startswith('Minor Class'):
            GName = int(l[13:-1])
        if l.startswith('Class'):
            q[int(l[7:-1])] = GName

    for e in edgesLines:
        u, v = ast.literal_eval(e)
        if u in q and v in q:
            edges += [(u, v)]

    return edges, q


def printIdsTex(P, onlyshorterList=True, Ids=None):
    if Ids is None:
        Ids = list(P.Ids[0][0]) + list(P.Ids[0][1])
    lines = []
    for Id in sorted(Ids, key=lambda x: str(x)):
        # if str(Id).startswith('NN'):
        lines += [r'\item ' + str(Id) + r' {\color{blue} new}']
        lines += [' ']
        if str(Id).startswith('NN'):
            strId = ''
            for i in Id.Ids:
                for f in i:
                    strId += f[0] + '(' + f[1:] + ')='
                strId = strId[:-1] + ', '
            lines += ['$' + strId[:-2] + '$']
        else:
            lines += ['$' + str(Id.Ids) + '$']
        lines += [' ']

        min, max = P.getIdBorders(Id)

        lines += [r'satisfied by (minimal ones): ' + str(min)]
        lines += [' ']
        lines += [r'not satisfied by (maximal ones): ' + str(max)]
        lines += [' ']

        satBy = [gid for gid in P.Graphs if Id in P.Ids[P.getCId(gid)][0]]
        nonsatBy = [gid for gid in P.Graphs if Id in P.Ids[P.getCId(gid)][1]]
        if onlyshorterList:
            if len(satBy) < len(nonsatBy):
                lines += [r'satisfied by: ' + str(satBy)]
            else:
                lines += [r'NOT satisfied by: ' + str(nonsatBy)]
            lines += [' ']
        else:
            lines += [r'satisfied by: ' + str(satBy)]
            lines += [' ']
            lines += [r'not satisfied by: ' + str(nonsatBy)]
            lines += [' ']

        restgids = [gid for gid in P.Graphs if not Id in P.Ids[P.getCId(gid)][0] and not Id in P.Ids[P.getCId(gid)][1]]
        if len(restgids) > 0:
            lines += [r'undetermined in: ' + str(restgids)]
            lines += [' ']

    for l in lines:
        print(l)


# ls=set([tuple([frozenset(P.Ids[c][0]),frozenset(P.Ids[c][1])]) for c in P.classes])
# goto https://www.wolframcloud.com/env/c5209d0e-8abb-487c-994d-800848d813d2

# ls = [(str(Id.Ids),str(Id),[gid for gid in P.Graphs if Id in P.Ids[P.getCId(gid)][0]]) for Id in sorted(Ids,key=lambda x: str(x))]
def getWolframGraph(P):
    lines = []
    G = Poset.removeTransitiveEdges(P.Graph.copy())
    t = [str(v) + '->' + str(u) for (u, v) in G.edges]
    graph = '{' + str(t).replace('\'', '')[1:-1] + '}'
    lines += ['g=' + graph]

    openG = nx.DiGraph()
    openG.add_nodes_from(G.nodes)
    openG.add_edges_from([(u, v) for u in openG.nodes for v in openG.nodes if
                          u != v and not P.Graph.has_edge(u, v) and not P.SeparationGraph.has_edge(u, v)])

    vertexLabels = 'VertexLabels -> {'
    for v in openG.nodes:
        vertexLabels += str(v) + '->\"' + str(P.classes[v])[1:-1] + '\",'
    vertexLabels = vertexLabels[:-1] + '}'

    green = [str(v) + '-> Green' for v in P.classes if openG.degree[v] == 0]
    orange = [str(v) + '-> Orange' for v in P.classes if openG.degree[v] > 0 and openG.degree[v] < 10]
    red = [str(v) + '-> Red' for v in P.classes if openG.degree[v] >= 10]
    vertexStyle = 'VertexStyle -> {' + str(green).replace('\'', '')[1:-1] + ',' + str(orange).replace('\'', '')[
                                                                                  1:-1] + ',' + str(red).replace('\'',
                                                                                                                 '')[
                                                                                                1:-1] + '}'
    print(lines[0])
    print('LayeredGraphPlot[g,VertexSize->0.35,' + vertexStyle + ',' + vertexLabels + ', ImageSize -> 1000]')


def getWolframInput(P):
    lines = []
    G = Poset.removeTransitiveEdges(P.Graph.copy())
    t = [str(v) + '->' + str(u) for (u, v) in G.edges]
    graph = '{' + str(t).replace('\'', '')[1:-1] + '}'
    lines += ['g=' + graph]
    for Id in sorted(list(P.Ids[0][0]) + list(P.Ids[0][1]), key=lambda x: str(x)):
        green = [str(v) + '-> Green' for v in P.classes if Id in P.Ids[v][0]]
        red = [str(v) + '-> Red' for v in P.classes if Id in P.Ids[v][1]]
        vertexStyle = 'VertexStyle -> {' + str(green).replace('\'', '')[1:-1] + ',' + str(red).replace('\'', '')[
                                                                                      1:-1] + '}'
        min, max = P.getIdBorders(Id)
        edgeStyle = 'EdgeStyle -> {'
        for (u, v) in G.edges:
            if u in {P.getCId(gid) for gid in max} or v in {P.getCId(gid) for gid in min}:
                edgeStyle += '(' + str(v) + '->' + str(u) + ')->Orange,'
        edgeStyle = edgeStyle[:-1] + '}'

        lines += [
            'CloudExport[LayeredGraphPlot[g,VertexSize->0.35,' + vertexStyle + ',' + edgeStyle + ', VertexLabels->Automatic, ImageSize -> 1000, PlotLabel -> \"' + str(
                Id) + '\"],\"PDF\",\"' + str(Id) + '\"]']
    for l in lines:
        print(l)


#    LayeredGraphPlot[g, VertexStyle -> { 4-> Green,  Directive[Black, 5], VertexLabels->Automatic, ImageSize -> 1000]

def getColoredOpenEdgeGraph(P: Poset.Poset):
    lines = []
    G = Poset.removeTransitiveEdges(P.Graph.copy())
    t = [str(v) + '->' + str(u) for (u, v) in G.edges]
    graph = '{' + str(t).replace('\'', '')[1:-1] + '}'
    lines += ['g=' + graph]
    openG = nx.DiGraph()
    openG.add_nodes_from(G.nodes)
    openG.add_edges_from([(u, v) for u in openG.nodes for v in openG.nodes if
                          u != v and not P.Graph.has_edge(u, v) and not P.SeparationGraph.has_edge(u, v)])
    print(set({openG.degree[v] for v in openG.nodes}))
    print(lines[0])
    vertexLabels = 'VertexLabels -> {'
    for v in openG.nodes:
        vertexLabels += str(v) + '->' + str(openG.degree[v]) + ','
    vertexLabels = vertexLabels[:-1] + '}'

    green = [str(v) + '-> Green' for v in P.classes if openG.degree[v] == 0]
    orange = [str(v) + '-> Orange' for v in P.classes if openG.degree[v] > 0 and openG.degree[v] < 10]
    red = [str(v) + '-> Red' for v in P.classes if openG.degree[v] >= 10]
    vertexStyle = 'VertexStyle -> {' + str(green).replace('\'', '')[1:-1] + ',' + str(orange).replace('\'', '')[
                                                                                  1:-1] + ',' + str(red).replace('\'',
                                                                                                                 '')[
                                                                                                1:-1] + '}'

    print('LayeredGraphPlot[g,VertexSize->0.35,' + vertexStyle + ',' + vertexLabels + ', ImageSize -> 1000]')

    return openG


def getIdGraph(P):
    Ids = list(P.Ids[0][0]) + list(P.Ids[0][1])
    G = nx.DiGraph()
    G.add_nodes_from(Ids)
    G.add_edges_from([(Id1, Id2) for Id1 in Ids for Id2 in Ids if
                      {cid for cid in P.classes if Id1 in P.Ids[cid][0]}.issubset(
                          {cid for cid in P.classes if Id2 in P.Ids[cid][0]})])
    return Poset.removeTransitiveEdges(G)


def getIdClasses(P):
    idss = set([(frozenset(P.Ids[cid][0]), frozenset(P.Ids[cid][1])) for cid in P.classes])
    classes = {ids: [gid for gid in P.Graphs if
                     (frozenset(P.Ids[P.getCId(gid)][0]), frozenset(P.Ids[P.getCId(gid)][1])) == ids] for ids in
               idss}

    print(sorted([sorted(classes[id]) for id in classes], key=lambda l: (len(l), max(l))))
    return classes


def printNewEdges(P):
    print([k for k in P.edges if not isinstance(P.edges[k], str)])
    t = str([str(k[0]) + '<' + str(k[1]) for k in P.edges if not isinstance(P.edges[k], str)])
    print(t.replace('\'', '').replace('<', '$<$'))


def addEqualityRelation(P):
    for i in P.Graphs:
        if {(v, v) for v in P.Graphs[i].nodes} not in [set(es) for es in P.Graphs[i].edges()]:
            P.Graphs[i] = Structures.Structure(P.Graphs[i].edges() + [{(v, v) for v in P.Graphs[i].nodes}])
    return P

#put intersections=False for segnificant speedup
def addCompositionsOfRelations(G, deleteUnnecessary=True, debug=False, returnNames=False,onlyLoopless=False,intersections=True):
    if isinstance(G, Poset.Poset):
        addEqualityRelation(G)
        for i in G.Graphs:
            G.Graphs[i] = addCompositionsOfRelations(G.Graphs[i], deleteUnnecessary=deleteUnnecessary, debug=debug)
        return
    workingEdges = [(set(es), str(i)) for i, es in enumerate(G.edges())]

    # foundNew = True
    edges = []
    while len(workingEdges) > 0:
        if debug: print(len(edges))
        newWorkingEdges = []
        # reverse
        newEdges = [({(v, u) for (u, v) in es}, 'R(' + name + ')') for es, name in workingEdges]
        for es, name in newEdges:
            if es not in [gs for gs, _ in edges + workingEdges + newWorkingEdges]:
                # edges += [(es,name)]
                newWorkingEdges += [(es, name)]
        # unary
        newEdges = [({(v, v) for (u, v) in es}, 'U(' + name + ')') for es, name in workingEdges]
        for es, name in newEdges:
            if es not in [gs for gs, _ in edges + workingEdges + newWorkingEdges]:
                # edges += [(es, name)]
                newWorkingEdges += [(es, name)]

        # compose
        for es, name in [({(u, w) for (u, v1) in es for (v2, w) in fs if v1 == v2}, '(' + n1 + 'o' + n2 + ')') for
                         es, n1 in workingEdges for fs, n2 in edges]:
            if es not in [gs for gs, _ in edges + workingEdges + newWorkingEdges] and len(es) > 0:
                if not onlyLoopless or len([(v,u) for (v,u) in es if u==v])==0:
                # edges += [(es, name)]
                    newWorkingEdges += [(es, name)]

        # intersection
        if intersections:
            for es, name in [({(u, v) for (u, v) in es if (u, v) in fs}, 'I(' + n1 + ',' + n2 + ')') for es, n1 in
                             workingEdges for fs, n2 in edges]:
                if es not in [gs for gs, _ in edges + workingEdges + newWorkingEdges] and len(es) > 0:
                    # edges += [(es, name)]
                    newWorkingEdges += [(es, name)]
        edges += workingEdges
        workingEdges = newWorkingEdges

    if debug: print('all computed',len(edges))
    # print(len(G.edges()),len(edges))
    if deleteUnnecessary:
        # delete reversed relations
        newEdges = []
        for es, name in edges:
            if not {(v, u) for (u, v) in es} in [gs for gs, _ in newEdges]:
                newEdges += [(es, name)]
        edges = newEdges
        if debug: print('deleted rev rel', len(edges))
        # delete intersections
        # TODO also consider reversed relations
        newEdges = []
        for es, name in edges:
            if not es in [{(u, v) for (u, v) in gs if (u, v) in fs} for gs, _ in newEdges for fs, _ in newEdges]:
                newEdges += [(es, name)]
        edges = newEdges
        if debug: print('deleted intersection rel', len(edges))
    if returnNames:
        return Structures.Structure([gs for gs, _ in edges]), edges
    return Structures.Structure([gs for gs, _ in edges])


def getH2s(n, elements=3, idAtTop=False):
    if idAtTop:
        tuples = list(itertools.product(list(range(elements)), repeat=n - 1))
        h2s = [{i: (i,) + t[i] for i in range(elements)} for t in list(itertools.product(tuples, repeat=elements))]
    else:
        tuples = list(itertools.product(list(range(elements)), repeat=n))
        h2s = [{i: t[i] for i in range(elements)} for t in list(itertools.product(tuples, repeat=elements)) if
               len(set(t)) == elements]
    # filter for same column
    h2sFiltered = []
    for h2 in h2s:
        rows = [tuple([h2[i][k] for i in range(elements)]) for k in range(n)]
        if len(set(rows)) == n:
            sorted = True

            for i in range(1 if idAtTop else 0, n - 1):
                if rows[i] > rows[i + 1]:
                    sorted = False
                    break
            if sorted:
                h2sFiltered += [h2]
    return h2sFiltered


def readNonFreeEdges(P=None):
    f = open('Non27edges.txt', 'r+')
    ls = f.readlines()
    f.close()
    if P is None:
        return [ast.literal_eval(l.split('pp-')[0]) for l in ls]
    es = []
    for l in ls:
        (u, v) = ast.literal_eval(l.split('pp-')[0])
        if P.getCId(u) != P.getCId(v) and not P.SeparationGraph.has_edge(P.getCId(u),
                                                                         P.getCId(v)) and not P.Graph.has_edge(
                P.getCId(u), P.getCId(v)):
            es += [(u, v)]
    return es


def tryFreeStructure(P, gid1Fix=None, gid2Fix=None, addAllBinRelations=False, timelimit=float('inf'), skipNonFree=True,
                     addFreePoly=False,ACWorks=None):
    h2 = {0: tuple([0] * 9 + [1] * 9 + [2] * 9), 1: tuple(([0] * 3 + [1] * 3 + [2] * 3) * 3), 2: tuple([0, 1, 2] * 9)}
    gid1s = P.Graphs
    skipEdges = []
    timeouts = {}
    if skipNonFree:
        skipEdges = readNonFreeEdges()
    if gid1Fix is not None:
        gid1s = [gid1Fix]
    for gid1 in gid1s:
        gid2s = P.Graphs
        if gid2Fix is not None:
            gid2s = [gid2Fix]
        for gid2 in gid2s:
            if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                    gid2)) and not P.SeparationGraph.has_edge(
                P.getCId(gid1), P.getCId(gid2)) and not (gid1, gid2) in skipEdges:
                G = P.Graphs[gid1]
                if addAllBinRelations:
                    G = addCompositionsOfRelations(G)
                Hedges = []
                stop=False
                for es in P.Graphs[gid2].edges():
                    if set(es) != {(0, 0), (1, 1), (2, 2)}:
                        Hedges += [es]
                        if gid1 in timeouts and frozenset(es) in timeouts[gid1]:
                            stop=True
                            break

                H = Structures.Structure(Hedges)
                print(gid1, gid2,G.edges(), H.edges())
                if stop:
                    print('timeout', gid1, timeouts[gid1])
                    continue
                d = Poset.getConstructionFromHomomorphisms(G, H, None, h2)
                if addFreePoly:
                    print(d.cs[0].G.edges())
                    d = addFreePolyToConstruction(d, G, H)
                    print(d.cs[0].G.edges())
                try:
                    # PG = d.apply(G,debug =True,timelimit=timelimit)
                    if ACWorks is None:
                        ACWorks = Poset.TSAllN in P.Ids[P.getCId(gid1)][0]
                    print('ACWorks', ACWorks)
                    PG = d.apply(G, ACWorks=ACWorks, debug=True, timelimit=timelimit)
                    #print('PG computed',len(PG.nodes),[(len(GG.nodes),len(GG.edges)) for GG in PG.Gs])
                    #print(Poset.ArcCon.existsHom(H, None, PG))
                    #print(Poset.ArcCon.existsHom(PG, ArcConFast.initF(PG, H,{str(h2[0]): {0}, str(h2[1]): {1}, str(h2[2]): {2}}),H))
                    if Poset.ArcCon.existsHom(PG, ArcConFast.initF(PG, H,
                                                                   {str(h2[0]): {0}, str(h2[1]): {1}, str(h2[2]): {2}}),
                                              H) and Poset.ArcCon.existsHom(H, None, PG):
                        P.addEdge(gid1, gid2, d)
                    else:
                        f = open('Non27edges.txt', 'a')
                        if addFreePoly:
                            f.writelines([str((gid1,
                                               gid2)) + ' pp-construction does not work with 27th power WITH existential quantifiers\n'])
                        else:
                            f.writelines([str((gid1,
                                               gid2)) + ' pp-construction does not work with 27th power without existential quantifiers\n'])
                        f.close()

                        # return Id
                        if gid1Fix is not None and gid2Fix is not None:
                            GPsmall = Poset.ArcCon.getMinimalNoHomEdgeSubstructure(PG, H)
                            return getFreeIdentity(GPsmall, H, 'FreeId(' + str(gid1) + ',' + str(gid2) + ')')

                except Exception as e:
                    if timelimit == float('inf'):
                        return
                    print('timeout', gid1,e)
                    print(H.Gs[int(str(e))].edges())
                    if gid1 not in timeouts:
                        timeouts[gid1]=set()
                    HE = H.Gs[int(str(e))]
                    isos = [{0:i,1:j,2:k} for i in range(3) for j in range(3) for k in range(3) if len({i,j,k})==3]
                    [timeouts[gid1].add(frozenset(nx.relabel_nodes(HE,isomorphism).edges())) for isomorphism in isos]

#todo only works for three element structures
def tryHomPPConstr(n, P: Poset.Poset, gid1s=None, h2s=None,random=0):
    goodH2s = []
    if h2s is None:
        tuples = list(itertools.product([0, 1, 2], repeat=n))
        h2s = [{0: a, 1: b, 2: c} for a in tuples for b in tuples for c in tuples if a != b and a != c and b != c]
    if random > 0:
        h2s = random.sample(tuples,random)
    if gid1s is None:
        gid1s = P.Graphs
    for gid1 in gid1s:
        for gid2 in P.Graphs:
            if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                    gid2)) and not P.SeparationGraph.has_edge(
                    P.getCId(gid1), P.getCId(gid2)):
                G = P.Graphs[gid1]
                Hedges = []
                for es in P.Graphs[gid2].edges():
                    if set(es) != {(0, 0), (1, 1), (2, 2)}:
                        Hedges += [es]
                H = Structures.Structure(Hedges)

                HWithEq = P.Graphs[gid2]
                print(gid1, gid2)
                # print(len(ds))
                for h2 in h2s:
                    d = Poset.getConstructionFromHomomorphisms(G, H, None, h2)
                    # print('ACWorks',Poset.TSAllN in P.Ids[P.getCId(gid1)][0])
                    PG = d.apply(G, ACWorks=Poset.TSAllN in P.Ids[P.getCId(gid1)][0])
                    if Poset.ArcCon.existsHom(PG, None, H) and Poset.ArcCon.existsHom(H, None, PG):
                        print(h2)
                        goodH2s += [h2]
                        d = Poset.getConstructionFromHomomorphisms(G, HWithEq, None, h2)
                        P.applyConstruction(d, applyTo=gid1)
                        break
    return goodH2s


#structuresComeFromDigraphs should be True if for every structure all relations can be pp-defined from its first relation
def tryRandomHomPPConstr(n, P: Poset.Poset, gid1s=None,structuresComeFromDigraphs=False,h2fix=None):
    if gid1s is None:
        gid1s = P.Graphs
    for gid1 in gid1s:
        for gid2 in P.Graphs:
            if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                    gid2)) and not P.SeparationGraph.has_edge(
                    P.getCId(gid1), P.getCId(gid2)):
                G = P.Graphs[gid1]
                if structuresComeFromDigraphs:
                    Hedges=[P.Graphs[gid2].edges()[0]]
                else:
                    Hedges = []
                    for es in P.Graphs[gid2].edges():
                        if set(es) != {(0, 0), (1, 1), (2, 2)}:
                            Hedges += [es]
                H = Structures.Structure(Hedges)

                HWithEq = P.Graphs[gid2]
                print(P.getNameOrId(gid1), P.getNameOrId(gid2))
                if h2fix is None:
                    h2 = dict()
                    for v in H.nodes:
                        h2[v]=random.choices(list(G.nodes),k=n)
                    print(h2)
                else:
                    h2=h2fix
                d = Poset.getConstructionFromHomomorphisms(G, H, None, h2)
                # print('ACWorks',Poset.TSAllN in P.Ids[P.getCId(gid1)][0])
                PG = d.apply(G, ACWorks=Poset.TSAllN in P.Ids[P.getCId(gid1)][0])
                #return PG
                if Poset.ArcCon.existsHom(PG, None, H) and Poset.ArcCon.existsHom(H, None, PG):
                    print(h2)
                    d = Poset.getConstructionFromHomomorphisms(G, HWithEq, None, h2)
                    P.applyConstruction(d, applyTo=gid1)
                    break



def tryHomPPConstrFast(n, P: Poset.Poset, gid1s=None, h2s=None):
    goodH2s = []
    if h2s is None:
        tuples = list(itertools.product([0, 1, 2], repeat=n))
        h2s = [{0: a, 1: b, 2: c} for a in tuples for b in tuples for c in tuples if a != b and a != c and b != c]

    if gid1s is None:
        gid1s = P.Graphs
    for gid1 in gid1s:
        G = P.Graphs[gid1]

        # gather all edge relations
        Hedges = set()
        oldNumberOfHEdges=0
        for gid2 in P.Graphs:
            if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                    gid2)) and not P.SeparationGraph.has_edge(
                    P.getCId(gid1), P.getCId(gid2)):
                Hedges.update(frozenset(es) for es in P.Graphs[gid2].edges())
                oldNumberOfHEdges += len(P.Graphs[gid2].edges())-1
        if frozenset({(0, 0), (1, 1), (2, 2)}) in Hedges:
            Hedges.remove(frozenset({(0, 0), (1, 1), (2, 2)}))
        if len(Hedges) > 0:
            print('Hedges wo dublicates', len(Hedges),'with dublicates', oldNumberOfHEdges, [list(es) for es in Hedges])
            HBig = Structures.Structure([list(es) for es in Hedges])

            for h2 in h2s:
                d = Poset.getConstructionFromHomomorphisms(G, HBig, None, h2)
                PGBigEges = d.apply(G).edges()
                HBigEdgesSets = [set(es) for es in HBig.edges()]

                for gid2 in P.Graphs:
                    if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                            gid2)) and not P.SeparationGraph.has_edge(
                            P.getCId(gid1), P.getCId(gid2)):
                        PGEdges = []
                        HEdges = []
                        HWithEq = P.Graphs[gid2]
                        for HE in HWithEq.edges():
                            if set(HE) != {(0, 0), (1, 1), (2, 2)}:
                                HEdges += [HE]
                                PGEdges += [PGBigEges[HBigEdgesSets.index(set(HE))]]
                        H = Structures.Structure(HEdges)
                        PG = Structures.Structure(PGEdges)
                        print(gid1, gid2)
                        # print(len(ds))
                        if Poset.ArcCon.existsHom(PG, None, H) and Poset.ArcCon.existsHom(H, None, PG):
                            print(h2)
                            goodH2s += [h2]
                            d = Poset.getConstructionFromHomomorphisms(G, HWithEq, None, h2)
                            P.applyConstruction(d, applyTo=gid1)

    return goodH2s

#todo there is a mistake here {0: (0, 0, 0), 1: (0, 1, 2), 2: (1, 1, 1)} add Edge 177 156
def tryHomPPConstrVeryFast(n, P: Poset.Poset, gid1s=None, h2s=None):
    goodH2s = []
    if h2s is None:
        tuples = list(itertools.product([0, 1, 2], repeat=n))
        h2s = [{0: a, 1: b, 2: c} for a in tuples for b in tuples for c in tuples if a != b and a != c and b != c]

    if gid1s is None:
        gid1s = P.Graphs
    for gid1 in gid1s:
        G = P.Graphs[gid1]

        # gather all edge relations
        Hedges = []
        oldNumberOfHEdges=0
        edgeId=dict()
        for gid2 in P.Graphs:
            if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                    gid2)) and not P.SeparationGraph.has_edge(
                    P.getCId(gid1), P.getCId(gid2)):
                oldNumberOfHEdges += len(P.Graphs[gid2].edges())-1
                for es in P.Graphs[gid2].edges():
                    if frozenset(es) not in edgeId and frozenset(es) != frozenset({(0, 0), (1, 1), (2, 2)}):
                        Hedges += [frozenset(es)]
                        isos = [{0: i, 1: j, 2: k} for i in range(3) for j in range(3) for k in range(3) if
                                len({i, j, k}) == 3]
                        for h in isos:
                            edgeId[frozenset({(h[u],h[v]) for (u,v) in es})]=(len(Hedges)-1,h)
                            #..add(frozenset(nx.relabel_nodes(HE, isomorphism).edges())) for isomorphism in isos]


        if frozenset({(0, 0), (1, 1), (2, 2)}) in Hedges:
            Hedges.remove(frozenset({(0, 0), (1, 1), (2, 2)}))
        if len(Hedges) > 0:
            print('Hedges wo dublicates', len(Hedges),'with dublicates', oldNumberOfHEdges, [list(es) for es in Hedges])
            HBig = Structures.Structure([list(es) for es in Hedges])
            #print(list(HBig.edges()))
            for h2 in h2s:
                d = Poset.getConstructionFromHomomorphisms(G, HBig, None, h2)
                PGBigEges = d.apply(G).edges()
                HBigEdgesSets = [set(es) for es in HBig.edges()]

                for gid2 in P.Graphs:
                    if P.getCId(gid1) != P.getCId(gid2) and not P.Graph.has_edge(P.getCId(gid1), P.getCId(
                            gid2)) and not P.SeparationGraph.has_edge(
                            P.getCId(gid1), P.getCId(gid2)):
                        PGEdges = []
                        HEdges = []
                        HWithEq = P.Graphs[gid2]
                        #print(list(HWithEq.edges()))
                        for HE in HWithEq.edges():
                            if set(HE) != {(0, 0), (1, 1), (2, 2)}:
                                HEdges += [HE]
                                #apply iso
                                id,h = edgeId[frozenset(HE)]
                                #print(HE,id,h,len(PGBigEges))
                                hInv = {h[i]:i for i in [0,1,2]}
                                PGEdge = []
                                for (u,v) in PGBigEges[id]:
                                    PGEdge +=[(str(tuple([hInv[ui] for ui in ast.literal_eval(u)])),str(tuple([hInv[vi] for vi in ast.literal_eval(v)])))]
                                PGEdges += [PGEdge]
                        H = Structures.Structure(HEdges)
                        PG = Structures.Structure(PGEdges)
                        print(gid1, gid2)
                        # print(len(ds))
                        if Poset.ArcCon.existsHom(PG, None, H) and Poset.ArcCon.existsHom(H, None, PG):
                            print(h2)
                            goodH2s += [h2]
                            d = Poset.getConstructionFromHomomorphisms(G, HWithEq, None, h2)
                            P.applyConstruction(d, applyTo=gid1)

    return goodH2s


def relToTikz(R):
    lines = [r'\begin{pmatrix}']
    a = ''
    b = ''
    for e in R:
        a += str(e[0]) + '&'
        b += str(e[1]) + '&'
    lines += [a[:-1] + r'\\', b[:-1]]
    lines += [r'\end{pmatrix},']
    return lines


def edgeToTik(P, gid, hid, colors=None):
    if colors is None:
        colors = ['green', 'red', 'dashed']
    G = P.Graphs[gid]
    H = P.Graphs[hid]
    e = (gid, hid, P.edges[(gid, hid)])
    lines = [r'\begin{lem}']
    lines += [r'$\mathcal C_{' + str(e[0]) + '}<\mathcal C_{' + str(e[1]) + '}$ with ']
    lines += [r'$\mathcal C_{' + str(e[0]) + '}=(']
    for GE in G.Gs:
        lines += relToTikz(GE.edges())
    # remove last comma
    lines[-1] = lines[-1][:-1]
    lines += [r')$ and']
    lines += [r'$\mathcal C_{' + str(e[0]) + '}=(']
    for HE in H.Gs:
        lines += relToTikz(HE.edges())
    # remove last comma
    lines[-1] = lines[-1][:-1]
    lines += [r')$']
    lines += [r'\end{lem}']
    for l in lines:
        print(l)

    print(colors)
    e[2].toTikz(colors=colors)

    GP = e[2].apply(G)
    h = Poset.ArcCon.findHom(H, GP)
    h2 = Poset.ArcCon.findHom(GP, H)
    print(r'$h\colon' + str([str(k) + '\\mapsto ' + str(h[k]) for k in h]).replace('\'', '') + '$')
    print(' ')
    print('$h\'\\colon' + str([str(k) + '\\mapsto ' + str(h2[k]) for k in h2]).replace('\'', '') + '$')


# hs={0:tuple([0]*9+[1]*9+[2]*9),1:tuple(([0]*3+[1]*3+[2]*3)*3),2:tuple([0,1,2]*9)}
# d=Poset.getConstructionFromHomomorphisms(G, H, None, hs)
# GP=d.apply(G)
# GP=ArcConFast.getMinimalNoHomSubgraphFast(GP,H,{str(hs[0]):{0},str(hs[1]):{1},str(hs[2]):{2}})
def getFreeIdentity(GP, H, name=None):
    fsNames = 'fghijklmnopqrstuvwxyzabcde'
    vs = list(GP.nodes)
    vTof = {vs[k]: fsNames[k] for k in range(len(vs))}
    hs = {0: tuple([0] * 9 + [1] * 9 + [2] * 9), 1: tuple(([0] * 3 + [1] * 3 + [2] * 3) * 3), 2: tuple([0, 1, 2] * 9)}
    fs = {fsNames[k]: {(hs[0][i], hs[1][i], hs[2][i]): ast.literal_eval(vs[k])[i] for i in range(27)} for k in
          range(len(vs))}
    identifications = [(i, j, k) for i in [0, 1, 2] for j in [0, 1, 2] for k in [0, 1, 2]]
    identificationsMaps = lambda tup: lambda k: (k[tup[0]], k[tup[1]], k[tup[2]])
    ids1=[]
    if False:
        minors = {
            (ident, f): {(i, j, k): fs[f][identificationsMaps(ident)((i, j, k))] for i in [0, 1, 2] for j in [0, 1, 2] for k
                         in [0, 1, 2]} for f in fs for ident in identifications}
        print('minors', len(minors))

        ids1 = [[k[1] + str(k[0][0]) + str(k[0][1]) + str(k[0][2]), l[1] + str(l[0][0]) + str(l[0][1]) + str(l[0][2])] for k
                in
                minors for l in minors if
                k != l and False not in [minors[k][(a, b, c)] == minors[l][(a, b, c)] for a in [0, 1, 2] for b in [0, 1, 2]
                                         for c in [0, 1, 2]]]
        # clean up ids1
        ids1New = []
        for id1 in ids1:
            added = False
            for id2 in ids1New:
                if id1[0] in id2:
                    if id1[1] not in id2:
                        id2 += [id1[1]]
                    added = True
                    break
                if id1[1] in id2:
                    if id1[0] not in id2:
                        id2 += [id1[0]]
                    added = True
                    break
            if not added:
                ids1New += [id1]
        ids1 = ids1New
    ids2 = []
    gsNames = [chr(i) for i in range(500, 10000)]
    g = 0
    for i, HE in enumerate(H.Gs):
        es = list(HE.edges())
        x = ''
        y = ''
        for e in es:
            x += str(e[0])
            y += str(e[1])
        for e in GP.Gs[i].edges():
            print(g, e)
            ids2 += [vTof[e[0]] + '012', gsNames[g] + x], [vTof[e[1]] + '012', gsNames[g] + y]
            g += 1

    return Identities.Identity(ids1+ids2, name)




def getTupleNumber(v):
    res=0
    for i in range(len(v)):
        res+=v[i]*(3**i)
    return res

# P=Poset.posetFromFile('TreeElementsWithColWithFree4')
# H=Structures.Structure(P.Graphs[109].edges()[:-1])
# G=P.Graphs[274]
# hs={0:tuple([0]*9+[1]*9+[2]*9),1:tuple(([0]*3+[1]*3+[2]*3)*3),2:tuple([0,1,2]*9)}
# d=Poset.getConstructionFromHomomorphisms(G, H, None, hs)
# d2=ThreeElements.addFreePolyToConstruction(d,G,H)
# GP=d2.apply(G)
def addFreePolyToConstruction(d: Poset.PPPowerBinarySignature, G, H,maxArity=7,onlyAddConnected=False):
    hs = {0: tuple([0] * 9 + [1] * 9 + [2] * 9), 1: tuple(([0] * 3 + [1] * 3 + [2] * 3) * 3), 2: tuple([0, 1, 2] * 9)}
    for i, c in enumerate(d.cs):
        Hes = list(H.Gs[i].edges())
        arity = len(Hes)
        print(i,arity)
        q=dict()
        if arity<=maxArity:
            # add = (last edge in edgelist is = edge)
            for vid in range(27):
                v = getTupleNumber(tuple([hs[Hes[l][0]][vid] for l in range(len(Hes))]))+100
                # print(vid,v,Hes,(hs[0][vid],hs[1][vid],hs[2][vid]))
                #c.G.Gs[-1].add_edge(vid, v)
                q[v]=vid
            for vid in range(27):
                v = getTupleNumber(tuple([hs[Hes[l][1]][vid] for l in range(len(Hes))]))+100
                # print(vid,v,Hes)
                #c.G.Gs[-1].add_edge(vid + 27, v)
                q[v]=vid+27
            if not onlyAddConnected:
                # c.G.add_nodes_from(itertools.product([0,1,2],repeat=arity))
                for j, GE in enumerate(G.Gs):
                    # add poly edges
                    productEdges = itertools.product(list(GE.edges()), repeat=arity)
                    for es in productEdges:
                        u = getTupleNumber(tuple([e[0] for e in es])) + 100
                        if u in q: u=q[u]
                        v = getTupleNumber(tuple([e[1] for e in es])) + 100
                        if v in q: v=q[v]
                        c.G.Gs[j].add_edge(u, v)
            else:
                addNewNodes = True
                es = list(c.G.edges())
                c.G = Structures.Structure(es)
                while addNewNodes:
                    addNewNodes=False
                    for j, GE in enumerate(G.Gs):
                        # add poly edges
                        productEdges = list(itertools.product(list(GE.edges()), repeat=arity))
                        #print('productEdges',len(productEdges))
                        for es in productEdges:
                            u = getTupleNumber(tuple([e[0] for e in es])) + 100
                            if u in q: u = q[u]
                            v = getTupleNumber(tuple([e[1] for e in es])) + 100
                            if v in q: v = q[v]
                            if not c.G.Gs[j].has_edge(u,v) and (v in c.G.Gs[j].nodes or u in c.G.Gs[j].nodes):
                                addNewNodes = True
                                c.G.Gs[j].add_edge(u, v)
                        es = list(c.G.edges())
                        c.G = Structures.Structure(es)
                        print(j,len(c.G.nodes),len(c.G.Gs[j].edges))

            es = list(c.G.edges())
            c.G = Structures.Structure(es)
        else:
            print('scipped relation because of high arity')
    return d

def cleanUpNoExFreePPPower(GP,G,phi,es=None):
    if es is None:
        es = list(GP.edges())
        #todo

def getSmallestUpperBounds(u,v,G):
    commonSuccs = set(G.successors(u)).intersection(set(G.successors(v)))
    commonSuccs2 = commonSuccs.copy()
    for v in commonSuccs2:
        if v in commonSuccs:
            commonSuccs.difference_update(G.successors(v))
            commonSuccs.add(v)
    return commonSuccs

def isLattice(G:nx.DiGraph):
    G=G.copy()
    G.add_edges_from([(v,v) for v in G.nodes]) # add reflexive edges
    vs = list(G.nodes)
    Grev = G.reverse()
    for i in range(len(vs)):
        for j in range(i+1,len(vs)):
            meet = getSmallestUpperBounds(vs[i],vs[j],G)
            join = getSmallestUpperBounds(vs[i],vs[j],Grev)
            if len(meet)>1 or len(join)>1:
                print('no lattice', vs[i],vs[j],'meet',meet,'join',join)
    print('is Lattice')


#use tryRandomHomPPConstr(P)
def getStructurePosetFromDigraphPoset(P,addComposition=True,addEdges=True,maxNumberOfElements=10**10):
    Q=Poset.Poset()
    if addComposition:
        for i in P.Graphs:
            if len(P.Graphs[i].nodes)<=maxNumberOfElements:
                G=addCompositionsOfRelations(Structures.Structure([list(P.Graphs[i].edges())]),intersections=False)
                print('number of relations:', len(G.edges()))
                Q.addGraph(G,  P.Names[i] if i in P.Names else None)
    else:
        [Q.addGraph(Structures.Structure([list(P.Graphs[i].edges())]),
                    P.Names[i] if i in P.Names else None) for i in P.Graphs]

    if addEdges:
        for e in P.edges:
            if  e[0] in P.Names and e[1] in P.Names:
                id1 = Q.getIdByName(P.Names[e[0]])
                id2 = Q.getIdByName(P.Names[e[1]])
                if id1 is not None and id2 is not None:
                    Q.addEdge(id1,id2, 'from parent poset')

    #[P.addIdentity(P.getCId(P.getIdByName(Q.Names[Q.classes[cid][0]])),True,id) for cid in Q.classes for id in Q.Ids[cid][0] if Q.classes[cid][0] in Q.Names and Q.Names[Q.classes[cid][0]] in P.Names ]
    return Q