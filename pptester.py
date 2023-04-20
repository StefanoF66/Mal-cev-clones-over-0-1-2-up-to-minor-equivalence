import itertools
import sys
from functools import reduce  # Valid in Python 2.6+, required in Python 3
import operator
import random

import Poset
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

import ArcConFast
import ArcConFast as ArcCon
from pyvis.network import Network
import copy
import time
import sympy
import ast
import os.path
# import Poset

import Identities
import Structures


def colorDrawGraph(G, text='', colors=None):
    if colors is None:
        colors = dict()
    nt = Network('700px', '1000px', directed=True, heading=text)

    for v in G.nodes:
        if v in colors:
            nt.add_node(str(v), color=colors[v], size=3)
        else:
            nt.add_node(str(v), size=3)
    for (u, v) in G.edges:
        nt.add_edge(str(u), str(v))

    nt.show_buttons(filter_=['physics', 'layout'])
    nt.show('nx.html')

    time.sleep(0.5)


def drawGraph(G, phi=("asd", "")):
    nt = Network('700px', '1000px', directed=True, heading=phi)
    relabel = {v: str(v) for v in G.nodes}
    GStr = nx.relabel_nodes(G, relabel)
    nt.from_nx(GStr)
    # nt.set_options()
    nt.show_buttons(filter_=['physics', 'layout'])
    nt.show('nx.html')

    time.sleep(0.5)


#    pos = nx.layout.spring_layout(G, k=0.25, iterations=20)
#    #pos = graphviz_layout(G, prog='dot')#twopi#circo
#    # draw loops
#    nx.draw(G,pos, with_labels=True)
#    nodes = nx.draw_networkx_nodes(G, pos, [n for n in G.nodes if (n, n) in G.edges], node_size=90,
#                                   node_color="red")

#    #    nx.draw_networkx_edges(G, pos, edgelist=G.edges, arrowstyle="<|-", style="dashed")
#    ax = plt.gca()
#    ax.set_axis_off()
#    #ax.set_title(str(phi))
#    #plt.savefig(r"C:\Users\flori\Documents\Idempotents\Graph" + toStr(f) + ".png")
#    plt.show()


# phi=["012a","23AA"]


# \tikzstyle{bullet}=[circle,fill,inner sep=0pt,minimum size=3pt]
# \usetikzlibrary{arrows}
# \def\scale{0.5}


def getCoreTrees(n, filterSomeForOrientation=False, cores=True):
    if cores:
        ls = open(r'tripolys_data-master\\' + str(n) + r'\cores.edges').readlines()
    else:
        ls = open(r'tripolys_data-master\\' + str(n) + r'\trees.edges').readlines()

    for l in ls:
        T = nx.DiGraph(ast.literal_eval(l))
        if filterSomeForOrientation:
            lvl = getLevelsOfBalancedGraph(T)
            i = 0
            height = max(v for v in lvl)
            while len(lvl[i]) == len(lvl[height - i]) and i < height / 2:
                i += 1
            if len(lvl[i]) >= len(lvl[height - i]):
                yield T
        else:
            yield T


def getNoMajorityCoreTrees(n, filterSomeForOrientation=False):
    ls = open(r'tripolys_data-master\\' + str(n) + r'\majority_n.edges').readlines()
    res = [nx.DiGraph(ast.literal_eval(l)) for l in ls]
    if filterSomeForOrientation:
        resNew = []
        for T in res:
            lvl = getLevelsOfBalancedGraph(T)
            i = 0
            height = max(v for v in lvl)
            while len(lvl[i]) == len(lvl[height - i]) and i < height / 2:
                i += 1
            if len(lvl[i]) >= len(lvl[height - i]):
                resNew += [T]
        res = resNew
    return res


def getOrientation(T):
    lvl = getLevelsOfBalancedGraph(T)
    i = 0
    height = max(v for v in lvl)
    while len(lvl[i]) == len(lvl[height - i]) and i < height / 2:
        i += 1
    if len(lvl[i]) > len(lvl[height - i]):
        return 0
    if len(lvl[i]) == len(lvl[height - i]):
        return 1
    return 2


def treesToTikz(Ts):
    header = r'''\tikzstyle{bullet}=[circle,fill,inner sep=0pt,minimum size=3pt]
\usetikzlibrary{arrows}
\def\scale{0.5}
\def\hdist{1cm}
\def\vdist{1cm}
'''
    text = r'''\begin{figure}
\centering'''
    for i, T in enumerate(Ts):

        text += treeToTikz(T, i + 1)
        if (i + 1) % 4 == 0:
            text += '\n\n' + r'\vspace{\vdist}' + '\n'
        else:
            text += r'\hspace{\hdist}' + '\n'
    text += r'''\caption{NP-hard trees with 20 vertices.}
\label{fig:my_label}
\end{figure}'''
    return header + '\n' + text


def treeToTikz(T, name,nodeStyle='bullet',putLabel=False):
    # print(name)
    levels = 2
    f = ArcCon.findHom(T, getPath('1' * levels))
    while f is None and levels <= len(T.nodes):
        levels += 1
        f = ArcCon.findHom(T, getPath('1' * levels))
    # print('edgelevels:', levels)
    if levels == len(T.nodes)+1:
        f = {v: 0 for v in T.nodes}
    text = r'\begin{tikzpicture}[scale=\scale]' + '\n'

    counter = {l: 0 for l in range(levels + 1)}
    levelsTxt = {l: '' for l in range(levels + 1)}
    for v in T.nodes:
        label = ''
        if putLabel:
            label = ', label={right:$'+str(v)+'$}'
        levelsTxt[f[v]] += r'\node['+nodeStyle+label+'] (' + str(v) + ') at (' + str(counter[f[v]]) + ',' + str(
            f[v]) + ') {};' + '\n'
        counter[f[v]] += 1
    for i in range(levels + 1):
        text += r'% Level ' + str(i) + '\n'
        text += levelsTxt[i]
    text += r'\path[->,>=stealth' + '\']' + '\n'
    for e in T.edges:
        text += '(' + str(e[0]) + ') edge (' + str(e[1]) + ')\n'
    text += ';\n'
    text += r'\node at (' + str((max([counter[j] for j in counter]) - 1) / 2) + ',-1) {Tree ' + str(name) + '};\n'
    text += r'\end{tikzpicture}' + '\n'
    # print(text)
    return text


# print(pptester.graphsToTikz(pptester.get4GraphsInOrder(Poset.posetFromFile('p4Dual'))))
def get4GraphsInOrder(P):
    # P=Poset.posetFromFile('p4Dual')
    Gs = [P.Graphs[i] for c in P.classes for i in P.classes[c]]
    NotHard = [G for G in Gs if Identities.satisfysIdentity(G, Identities.Sigg3)]
    HardNotSmooth = [G for G in Gs if not Identities.satisfysIdentity(G, Identities.Sigg3) and not isSmooth(G)]
    HardSmooth = [G for G in Gs if not Identities.satisfysIdentity(G, Identities.Sigg3) and isSmooth(G)]
    return NotHard + HardNotSmooth + HardSmooth


def upToIsomorphism(Gs):
    Ugs = set()
    Gs = set(Gs)
    while len(Gs) > 0:
        G = Gs.pop()
        Ugs.add(G)
        Gs = {H for H in Gs if not nx.is_isomorphic(H, G)}

    #        print(len(Ugs))
    return Ugs


def graphsToTikz(Ts, columns=8):
    alph = 'ABCDEFGHIJKLMN'
    header = r'''\tikzstyle{bullet}=[circle,fill,inner sep=0pt,minimum size=3pt]
\usetikzlibrary{arrows}
\def\scale{0.5}
\def\hdist{1mm}
\def\vdist{3mm}
'''
    text = r'''\begin{figure}
\centering'''
    for i, T in enumerate(Ts):
        header += r'\def\G' + alph[i // 10] + alph[i % 10] + '{' + '\n'
        header += graphWithAtMost4VerticesToTikz(T, i + 1)
        header += '}\n'
        text += r'\G' + alph[i // 10] + alph[i % 10] + '\n'
        if (i + 1) % columns == 0:
            text += '\n \n' + r'\vspace{\vdist}' + '\n'
        else:
            text += r'\hspace{\hdist}' + '\n'
    text += r'''\caption{All core digraphs with at most 4 vertices.}
\label{fig:my_label}
\end{figure}'''
    return header + '\n' + text


def graphWithAtMost4VerticesToTikz(T, name):
    # print(name)
    text = r'\begin{tikzpicture}[scale=\scale]' + '\n'
    text += r'\clip(-0.8,-1.5) rectangle (1.8,1.2);' + '\n'
    pos = ['(0,0)', '(0,1)', '(1,1)', '(1,0)']
    for v in T.nodes:
        text += r'\node[bullet] (' + str(v) + ') at ' + pos[v] + '{};\n'
    text += r'\path[->,>=stealth' + '\']' + '\n'
    for e in T.edges:
        text += '(' + str(e[0]) + ') edge (' + str(e[1]) + ')\n'
    text += ';\n'
    text += r'\node at (0.5,-1) {Tree ' + str(name) + '};\n'
    text += r'\end{tikzpicture}' + '\n'
    # print(text)
    return text


def pathToNumbers(P: nx.DiGraph, copy=True):
    if copy:
        P = P.copy()
    if 0 in P.nodes and P.degree[0] == 1:
        v = 0
    else:
        vs = [v for v in P.nodes if P.degree[v] == 1]
        v = vs[0]
    w = ''
    while len(P.nodes) > 1:
        # print(w,v,list(P.neighbors(v)),list(P.edges))
        l = list(P.successors(v))
        if len(l) == 1:
            u = l[0]
            w += '1'
        else:
            u = list(P.predecessors(v))[0]
            w += '0'
        P.remove_node(v)
        v = u
    wOther = w[::-1].replace('0', 'a')
    wOther = wOther.replace('1', '0')
    wOther = wOther.replace('a', '1')
    if w < wOther:
        return w
    return wOther


def printTikzTableFromNumber(numbers):

    print(r'\pgfplotstableread{ % Read the data into a table macro')
    maxHM = max(numbers[max(numbers.keys())].keys())
    labels = 'Label '
    for i in range(maxHM):
        labels += str(i+1) + ' '
    print(labels)

    for i in range(1,max(numbers.keys())+1):
        text = str(i) + ' '
        summe = sum([numbers[i][k] for k in numbers[i]])
        for k in range(1,maxHM+1):
            if k in numbers[i]:
                anteil = str((numbers[i][k]*10**5)//summe)
                anteil = '0'*(6-len(anteil)) + anteil
                text += anteil[0]+'.'+anteil[1:] + ' '
            else:
                text += '0.00000 '
        print(text)
    print(r'''}\testdata

\begin{tikzpicture}
\begin{axis}[
            ybar stacked,   % Stacked horizontal bars
            ymin=0,         % Start x axis at 0
            ymax = 1,
            xtick=data,     % Use as many tick labels as y coordinates
            xticklabels from table={\testdata}{Label}, width=14cm, height = 7cm,line width=0.7pt,bar width=0.45cm
]

\def\plotcommand#1{
    \addplot [fill=black!#1!blue] table [y=\s, meta=Label,x expr=\coordindex] {\testdata};
}

\foreach \s in {1,...,14}
{
\pgfmathparse{ln(\s)*100/3}

\expandafter\plotcommand\expandafter{\pgfmathresult}
}
\addplot [fill=orange!70!white] table [y=30, meta=Label,x expr=\coordindex] {\testdata};
\end{axis}
\end{tikzpicture}''')

def computeHMTableForPathsUpTo(n,printTikz=True):
    start = time.time()
    k = 1
    numbers = dict()
    paths = dict()
    while k <= n and os.path.exists(r'pathsHM\\' + str(k)):

        numbers[k] = dict()
        f = open(r'pathsHM\\' + str(k))
        for line in f:
            l = line.rstrip().split(',')
            p = tuple([int(m) for m in l[0].split(' ')])
            mmin = int(l[1])
            paths[p] = mmin
            if mmin not in numbers[k]:
                numbers[k][mmin] = 0
            numbers[k][mmin] += 1 if 'True' == l[-1].replace(' ', '') else 2

        k += 1

    while k <= n:
        numberOfCorePaths = 0
        numbers[k] = dict()
        f = open(r'pathsHM\\' + str(k), 'w+')
        ps = getCorePathsFromFile(k)
        for p in ps:
            numberOfCorePaths += 1
            # print('k',k,len(ps),p.edges)
            w = pathToNumbers(p)
            # up to flipping
            if w > w[::-1]:
                continue

            wtxt, wshort = zopathToNumberpath(w, ' ', True)
            wtxt = wtxt[:-1]
            # print(wtxt,wshort)
            if '1' not in w:
                HM = 1
                mmin = 1
            else:
                # wshort = [int(l) for l in zopathToNumberpath(w,separator=' ')[:-1].split(' ')]
                if 1 not in wshort:
                    m = min(wshort)
                    wshort = [l - m + 1 for l in wshort]
                    mmin = paths[tuple(wshort)]
                else:
                    # simple test for NL-hard
                    # mmin=0
                    # for i in range(len(wshort)-4):
                    #    if wshort[i+2]< wshort[i+1] and wshort[i+2]< wshort[i+3]:
                    #        middle = wshort[i+1] - wshort[i+2]+ wshort[i+3]
                    #        if middle<  wshort[i] and middle < wshort[i+4]:
                    #            mmin=30
                    #            maxLength=0
                    #            mminpp=0
                    #            break

                    mmin = 0
                    for i in range(len(wshort) - 4):
                        if wshort[i] >= 3 and wshort[i + 1] < wshort[i] and wshort[i + 2] <= wshort[i + 1]:
                            maxHeight = wshort[i]
                            currentHeight = 0
                            currentMaxHeight = 0
                            path = []
                            sign = 1
                            for j in range(i + 1, len(wshort)):
                                newCurrentHeight = currentHeight + sign * wshort[j]
                                sign = sign * (-1)
                                if newCurrentHeight > maxHeight:
                                    break
                                if newCurrentHeight <= 0:
                                    if currentHeight == currentMaxHeight and currentHeight not in path:
                                        path += [currentHeight]
                                        if path != path[::-1]:
                                            # print('NLpath',path,i,wshort)
                                            mmin = 30
                                            maxLength = 0
                                            mminpp = 0
                                    if newCurrentHeight < 0:
                                        break
                                currentHeight = newCurrentHeight
                                currentMaxHeight = max(currentMaxHeight, currentHeight)
                                path += [wshort[j]]
                            if mmin > 0:
                                break

                    if mmin == 0:
                        # search for k' k k ... k k' pattern (k'>=k) and set min to maxPatternLength - 1
                        maxLength = 0
                        length = 1
                        kk = wshort[0]
                        for i in range(1, len(wshort)):
                            if wshort[i] == kk:
                                length += 1
                            elif wshort[i] > kk:
                                maxLength = max(maxLength, length + 1)
                                length = 1
                                kk = wshort[i]
                            else:  # wshort[i] < k
                                maxLength = max(maxLength, length)
                                length = 2
                                kk = wshort[i]
                        # print(wshort,maxLength)

                        # skipp 1 pp construction no speedup up to 15 vertices :(
                        wshortSkip1 = []
                        kk = 0
                        sameDirection = False
                        for i in range(len(wshort)):
                            if wshort[i] == 1:
                                sameDirection = not sameDirection
                            else:
                                if sameDirection:
                                    kk = kk - 1 + wshort[i]
                                    sameDirection = not sameDirection
                                else:
                                    wshortSkip1 += [kk]
                                    kk = wshort[i]
                                    sameDirection = False
                        wshortSkip1 = wshortSkip1[1:] + [kk]
                        # print(wshort,wshortSkip1)
                        m = min(wshortSkip1)
                        wshortSkip1 = [l - m + 1 for l in wshortSkip1]
                        if tuple(wshortSkip1) in paths:
                            mminpp = paths[tuple(wshortSkip1)]
                        else:
                            mminpp = 0

                        mmin = max(mminpp, maxLength - 1)

                        lvl = getLevelsOfBalancedGraph(p)
                    # print(wshort,maxLength)
                    # mmin = 2
                    mmax = 29
                    HM = max(4, mmin)
                    while mmin < mmax:
                        # print(mmin,HM,mmax)
                        if Identities.satisfysIdentity(p, Identities.getHM(HM), True, partition=lvl):
                            mmax = HM - 1
                        else:
                            mmin = HM + 1

                        if mmax == 29:
                            if mmin > maxLength - 1 + 2:
                                HM = mmax
                            else:
                                HM += 1
                        else:
                            HM = (mmin + mmax) // 2
                    paths[tuple(wshort)] = mmin
                    if maxLength - 1 != mmin and maxLength != 0:
                        print('L', maxLength - 1, 'pp', mminpp, 'mmin', mmin, wshort)
            f.writelines([wtxt + ', ' + str(mmin) + ', ' + str(w == w[::-1]) + '\n'])
            if mmin not in numbers[k]:
                numbers[k][mmin] = 0
            numbers[k][mmin] += 1 if w == w[::-1] else 2
        numbers[k][30]=numberOfCorePaths-sum([numbers[k][i] for i in numbers[k]])#number of NLhard paths
        k += 1
        f.close()
    print(time.time() - start)
    if printTikz:
        printTikzTableFromNumber(numbers)
    return numbers


def fastNLHardTest(wshort):
    for i in range(len(wshort) - 4):
        if wshort[i] >= 3 and wshort[i + 1] < wshort[i] and wshort[i + 2] <= wshort[i + 1]:
            maxHeight = wshort[i]
            currentHeight = 0
            currentMaxHeight = 0
            path = []
            sign = 1
            for j in range(i + 1, len(wshort)):
                print(j, path, currentHeight)
                newCurrentHeight = currentHeight + sign * wshort[j]
                sign = sign * (-1)
                if newCurrentHeight > maxHeight:
                    break
                if newCurrentHeight <= 0:
                    if currentHeight == currentMaxHeight and currentHeight not in path:
                        path += [currentHeight]
                        if path != path[::-1]:
                            print('NLpath', path, i, wshort)
                            return True
                    if newCurrentHeight < 0:
                        break
                currentHeight = newCurrentHeight
                currentMaxHeight = max(currentMaxHeight, currentHeight)
                path += [wshort[j]]
            return False


def computeHMTableForTreesUpTo(n, append=False, appendafter=0):
    k = 2
    numbers = dict()
    paths = dict()
    while k <= n and os.path.exists(r'treesHM\\' + str(k)):

        numbers[k] = dict()
        f = open(r'treesHM\\' + str(k))
        for line in f:
            l = line.rstrip().split(',')
            mmin = int(l[-2])
            if mmin not in numbers[k]:
                numbers[k][mmin] = 0
            numbers[k][mmin] += 1 if 'True' == (l[-1].replace(' ', '')) else 2
        if append and k >= appendafter:
            # todo numberofTrees is too large as it is not clear whether dual tree has been skipped
            numberOfTrees = sum([numbers[k][mm] for mm in numbers[k]])
            ps = getCoreTrees(k)
            numOfTrees = len(open(r'tripolys_data-master\\' + str(k) + r'\edges').readlines())
            if numberOfTrees < numOfTrees:
                print('append', k, numberOfTrees, numOfTrees, appendafter)
                f.close()
                ps = getCoreTrees(k)
                for i in range(numberOfTrees):
                    next(ps)
                f = open(r'treesHM\\' + str(k), 'a')
                for p in ps:
                    lvl = getLevelsOfBalancedGraph(p)
                    o = getOrientation(p)
                    if o == 2:
                        continue

                    mmin = 2
                    max = 29
                    HM = 4
                    while mmin < max:
                        if Identities.satisfysIdentity(p, Identities.getHM(HM), True, partition=lvl):
                            max = HM
                        else:
                            mmin = HM + 1
                        if mmin > 18:  # todo remove later
                            mmin = max
                        HM = (mmin + max) // 2

                    f.writelines([str(list(p.edges)) + ', ' + str(mmin) + ', ' + str(o == 1) + '\n'])
                    if mmin not in numbers[k]:
                        numbers[k][mmin] = 0
                    numbers[k][mmin] += 1 if o == 1 else 2

        k += 1

    while k <= n:
        numbers[k] = dict()
        f = open(r'treesHM\\' + str(k), 'w+')
        ps = getCoreTrees(k)
        for p in ps:

            lvl = getLevelsOfBalancedGraph(p)
            o = getOrientation(p)
            if o == 2:
                continue

            mmin = 2
            max = 29
            HM = 4
            while mmin < max:
                if Identities.satisfysIdentity(p, Identities.getHM(HM), True, partition=lvl):
                    max = HM
                else:
                    mmin = HM + 1
                HM = (mmin + max) // 2

            f.writelines([str(list(p.edges)) + ', ' + str(mmin) + ', ' + str(o == 1) + '\n'])
            if mmin not in numbers[k]:
                numbers[k][mmin] = 0
            numbers[k][mmin] += 1 if o == 1 else 2
        k += 1
        f.close()
    return numbers


def printTable(numbers):
    lines = [''] * 30
    for k in numbers:
        for i in range(1, 30):
            if i in numbers[k]:
                lines[i - 1] += str(numbers[k][i]) + ','
            else:
                lines[i - 1] += '0,'
    for l in lines:
        print(l)


def evaluate(G, phi):
    if (phi[0] + phi[1]).isnumeric():
        return [phi]
    res = []
    for l in "abcdefghijklmnopqrstuvwxyz":
        if l in phi[0] + phi[1]:
            break

    if l.capitalize() in phi[0] + phi[1]:
        for (u, v) in G.edges:
            res += evaluate(G, (phi[0].replace(l, str(u)).replace(l.capitalize(), str(v)),
                                phi[1].replace(l, str(u)).replace(l.capitalize(), str(v))))
    else:
        for v in G.nodes:
            res += evaluate(G, (phi[0].replace(l, str(v)), phi[1].replace(l, str(v))))
    return res


def evaluateB3(G, phi):
    R = phi[2]
    for r in R:
        if (phi[0] + phi[1])[r[0]] == '0' and (phi[0] + phi[1])[r[1]] == '0' and (phi[0] + phi[1])[r[2]] == '0':
            return []

    if (phi[0] + phi[1]).isnumeric():
        return [phi[:2]]

    res = []

    for l in "abcdefghijklmnopqrstuvwxyz":
        if l in phi[0] + phi[1]:
            break

    # if l.capitalize() in phi[0] + phi[1]:
    #    for (u, v) in G.edges:
    #        res += evaluate(G, (phi[0].replace(l, str(u)).replace(l.capitalize(), str(v)),
    #                            phi[1].replace(l, str(u)).replace(l.capitalize(), str(v))))
    # else:
    for v in G.nodes:
        res += evaluateB3(G, (phi[0].replace(l, str(v)), phi[1].replace(l, str(v)), R))
    return res


def pppower(G, phi):
    if len(G.nodes) > 10:
        print("Warning: too many nodes", phi, G.edges)
    if not isinstance(G, nx.DiGraph):
        Hs = []
        for i in range(len(G.Gs)):
            Hs += [pppower(G.Gs[i], phi[i])]
        return Structures.Structure(Hs)
    else:
        H = nx.DiGraph()
        edges = evaluate(G, phi)
        # print(edges)
        H.add_edges_from(edges)
        return H


def getLevelsOfBalancedGraphSlow(T):
    levels = 2
    f = ArcCon.findHom(T, getPath('1' * levels))
    while f is None:
        levels += 1
        f = ArcCon.findHom(T, getPath('1' * levels))

    lvl = dict()
    for i in range(levels + 1):
        lvl[i] = {v for v in f if f[v] == i}
    return lvl


def getLevelsOfBalancedGraph(T):
    nodes = list(T.nodes)
    levels = 2 * len(nodes)
    f = ArcCon.arcCon(T, {nodes[0]: {levels // 2}}, getPath('1' * levels), workingSet={nodes[0]})[0]

    lvl = dict()
    for i in range(levels + 1):
        lvl[0] = {v for v in f if f[v] == {i}}
        if len(lvl[0]) > 0:
            s = i
            break
    for i in range(s + 1, levels + 1):
        lvl[i - s] = {v for v in f if f[v] == {i}}
        if len(lvl[i - s]) == 0:
            del lvl[i - s]
            break

    return lvl


def pppowerB3(G, phi):
    H = nx.DiGraph()
    edges = evaluateB3(G, phi)
    # print(edges)
    H.add_edges_from(edges)
    return H


def relabelGraph(G):
    nodes = list(G.nodes)
    relabel = {nodes[i]: i for i in range(len(nodes))}
    return nx.relabel_nodes(G, relabel)


def getWords(n, alph):
    if n == 0:
        return [""]
    return [w + l for w in getWords(n - 1, alph) for l in alph]


def getTuples(n, alph):
    if n == 0:
        return [()]
    return [w + (l,) for w in getTuples(n - 1, alph) for l in alph]


def concat(ls):
    if len(ls) == 0:
        return ""
    return ls[0] + concat(ls[1:])


def getLowerNoDublicates(w):
    w = [a for a in w if a.islower()]
    res = ''
    for a in w:
        if a not in res:
            res += a
    return res


def getOrderedWords(alph, n, inc):
    if n == 0:
        return [""]
    ws = []
    for i in range(min(inc + 1, len(alph))):
        ws += [alph[i] + w for w in getOrderedWords(alph[i:], n - 1, inc)]
    return ws


def getFormulaFirstTuple(vs, n):
    fst = [getOrderedWords("abcdefghijklmnopqrstuvwxyz", k, 1) for k in range(n)]
    snd = [getOrderedWords("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n], k, n) for k in range(n)]
    trd = [getOrderedWords(vs, k, len(vs)) for k in range(n)]

    ws = []
    for i in range(0, n):
        for j in range(0, n - i):
            ws += ['a' + u + v + w for u in fst[i] for v in snd[j] for w in trd[n - j - i - 1]]
    return ws


def getDiCycle(n):
    return nx.DiGraph([(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)])


def isUnionOfCycles(G: nx.DiGraph):
    for v in G.nodes:
        if G.in_degree[v] != 1:
            return False
        if G.out_degree[v] != 1:
            return False
    return True


def isUnionOfCyclesSlow(G):
    Gs = nx.weakly_connected_components(G)
    for subG in Gs:
        if not isCycle(G.subgraph(subG)):
            return False
    return True


def isCycle(G):
    return nx.is_isomorphic(getDiCycle(len(G.nodes)), G)


def isSmooth(G):
    for v in G.nodes:
        if G.in_degree[v] == 0 or G.out_degree[v] == 0:
            return False
    return True


# for tikzing
def whichTreesAreAlmostTheSame(Ts):
    TTs = []
    for T in Ts:
        l = getLevelsOfBalancedGraph(T)
        if len(l[0]) == 1:
            v = l[0].pop()
            T = T.copy()
            T.remove_node(v)
        if len(l[max(l.keys())]) == 1:
            v = l[max(l.keys())].pop()
            T = T.copy()
            T.remove_node(v)
        TTs += [T]
    for i in range(len(TTs)):
        for j in range(i):
            if nx.is_isomorphic(TTs[i], TTs[j]) or nx.is_isomorphic(TTs[i], TTs[j].reverse()):
                print(i, '=', j)


def getUsualFormulas(maxVar=10, minVar=0):
    fs = [('0', '0'), ('A', 'a'), ('0', '1'), ('ab', 'bA'), ('aA', 'ba'), ('aA', 'Aa'), ('aBC', 'bcA')]  # TODO add more
    fs += [('a0', '1A'),
           ('a0', '2A'),
           ('a0', 'A1'),
           ('a0', 'AA'),
           ('a1', '1A'),
           ('a1', '2A'),
           ('a1', '3A'),
           ('a2', '3A'),
           ('a3', '3A'),
           ('aA', 'Aa'),
           ('aA', 'ba'),
           ('aa', 'A1'),
           ('aa', 'A2'),
           ('aa', 'A3'),
           ('ab', 'bA')]
    fs += [('abb', 'CAc'),
           ('aab', '2BA'),
           ('aAB', 'b2a'),
           ('aB0', 'b2a'),
           ('aB2', 'b2a'),
           ('aBC', 'bcA')]

    fs += [('abbA', 'CBc2'), ('aabC', '3Bc0'), ('abbc', 'baCA')]

    fs += [('aabC', '3Bc0'),
           ('abbc', 'baCA'),
           ('abc0', 'CACb'),
           ('abbc', 'Ca3B'),
           ('abc1', 'CACb'),
           ('abbA', 'CBc2'),
           ('abc2', 'CACb'),
           ('abc3', 'CACb'),
           ('abc2', '2aBC')]

    fs += [('abcd1', 'B0A1c'), ('abcCC', 'ABBdA')]

    fs += [('0', '0'),
           ('0', '1'),
           ('A', 'a'),
           ('a0', '1A'),
           ('a0', '2A'),
           ('a0', '3A'),
           ('a1', '1A'),
           ('a1', '2A'),
           ('a1', '3A'),
           ('a2', '2A'),
           ('a2', '3A'),
           ('a3', '2A'),
           ('aA', 'ba'),
           ('aAB', '0bb'),
           ('aAB', 'b2a'),
           ('aAB', 'b2b'),
           ('aAB', 'b3b'),
           ('aB0', 'b2a'),
           ('aB1', '0Ab'),
           ('aB2', 'b2a'),
           ('aB3', 'bbA'),
           ('aBC', 'bcA'),
           ('aa', 'A1'),
           ('aa', 'A2'),
           ('aa', 'A3'),
           ('aab', '2BA'),
           ('aab', 'AB0'),
           ('aabC', '3Bc0'),
           ('ab', 'bA'),
           ('ab2', '1AB'),
           ('abA', '0AB'),
           ('abA', 'BCc'),
           ('abB', 'A1A'),
           ('abb', 'CAc'),
           ('abbA', 'CBc2'),
           ('abbc', 'Ca3B'),
           ('abbc', 'baCA'),
           ('abc0', 'CACb'),
           ('abc1', 'CACb'),
           ('abc2', '2aBC'),
           ('abc2', 'CACb'),
           ('abc3', 'CACb'),
           ('abcCC', 'ABBdA'),
           ('abcd1', 'B0A1c'), ]
    fs += [('0ABC', 'abc2'), ('1ABC', 'abc3'), ('0aBC', 'Abc2'), ('2aBC', 'Abc1'), ('2aBC', 'Abc2'), ('4AbC', 'aBc2'),
           ('1ABcD', 'abCd1'), ('0aBcD', 'AbCd2'), ('0aBcD', 'AbCd1'), ('abBCD', 'BcdaC')]
    fs += [('aab', 'bAc')]  # T4 <= N321T3
    fs += [('3aBCD', 'Abcd3')]
    fs += [('aAB', '0bb'),
           ('abb', 'caC'),
           ('abc3', 'CACb'),
           ('aA', 'ba'),
           ('aBC', 'bcA'),
           ('abc', 'bB5'),
           ('0', '0'),
           ('aB0', 'b2a'),
           ('aB3', '76b'),
           ('abb', 'CAc'),
           ('abD2', 'AdbD'),
           ('ab', 'bA'),
           ('aB2', 'b2a'),
           ('a1', '1A'),
           ('A', 'a'),
           ('a3', '2A'),
           ('aAB', 'b2a'),
           ('a4', '1A'),
           ('abc1', 'CACb'),
           ('aab', '7B2'),
           ('ab2', '1AB'),
           ('aB2', '5bA'),
           ('abc2', 'CACb'),
           ('aB', 'Ab'),
           ('aB1', '76b'),
           ('aab', '2BA'),
           ('a2', '3A'),
           ('a0', 'AA'),
           ('abcCC', 'ABBdA'),
           ('abc', 'BAb')]
    # [(str(i)+'a','A'+str(j)) for i in range(10) for j in range(10)]
    return list({f for f in fs if numberOfVariables(f) <= maxVar and numberOfVariables(f) >= minVar})


def numberOfVariables(phi):
    vs = {a for a in phi[0] + phi[1] if a.islower()}
    return len(vs)


def zopathToNumberpath(w, separator='', returnList=False):
    if w == '':
        if not returnList:
            return ''
        return ('', [])
    n = w[0]
    i = 1
    while i < len(w) and w[i] == n:
        i += 1
    if not returnList:
        return str(i) + separator + zopathToNumberpath(w[i:], separator)
    txt, l = zopathToNumberpath(w[i:], separator, returnList)
    return (str(i) + separator + txt, [i] + l)


def getT3Formulas(n):
    return [(str(i) + 'a', 'A' + str(j)) for i in range(n) for j in range(n)]


def getTriad(w1, w2, w3):
    G1 = getPath(w1)
    G2 = getPath(w2)
    G3 = getPath(w3)
    nx.relabel_nodes(G2, lambda v: 0 if v == 0 else v + len(w1), False)
    nx.relabel_nodes(G3, lambda v: 0 if v == 0 else v + len(w1) + len(w2), False)
    G1.add_edges_from(G2.edges)
    G1.add_edges_from(G3.edges)
    return G1


# todo does not work
def getTriads(n):
    ps = []
    for i in range(1, n + 1):
        ps += getRootedTwoBraidedPaths(i)
    res = []
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            T = ps[i].copy()
            S = ps[j].copy()
            relab = {v: str(v) + 'b' for v in S.nodes}
            relab[0] = 0
            S = nx.relabel_nodes(S, relab)
            T.add_edges_from(S.edges)
            if ArcCon.isTreeCore(T, {0: {0}}):
                for k in range(j + 1, len(ps)):
                    S = ps[k].copy()
                    relab = {v: str(v) + 'c' for v in S.nodes}
                    relab[0] = 0
                    S = nx.relabel_nodes(S, relab)
                    T.add_edges_from(S.edges)
                    if ArcCon.isTreeCore(T, {0: {0}}):
                        res += [T]
    return res


def getTnFormulas(n, const=None):
    if const is None:
        const = [0, 1, 2, 3]
    alph = 'abcdefghijklmnopqrstuvwxyz'
    res = []
    for w in getWords(n - 1, '01'):
        a = ''
        b = ''
        for i in range(n - 1):
            if w[i] == 0:
                a += alph[i].upper()
                b += alph[i]
            else:
                a += alph[i]
                b += alph[i].upper()

        res += [(a + str(i), str(j) + b) for i in const for j in const]
    return res


# phi is of the form "variables constants","..." both in increasing order in the first tuple
def getReasonableFormulas(G, n, skip=1, k=None):
    if k == None:
        k = n
    vs = concat([str(v) for v in G.nodes])
    # vs = "03" #N5
    # vs = "04" #N7

    # at least one variable a somewhere wlog at first position
    fstws = getFormulaFirstTuple(vs, n)
    if k == 0:
        fstws = [w for w in fstws if 'A' not in w and 'B' not in w and 'C' not in w and 'D' not in w]
    print("fst tuple: " + str(len(fstws)))
    sndws = getWords(n, vs + "abcdefghijklmnopqrstuvwxyz"[:n] + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:k])
    random.shuffle(sndws)
    sndws = sndws[::skip]
    # TODO n-1 back to n
    print("snd tuple: " + str(len(sndws)))

    phis = []
    for w2 in sndws:
        # not only constants
        if w2.isnumeric():
            continue
        v2 = getLowerNoDublicates(w2)
        for w1 in fstws:
            use = True
            w = w1 + w2
            containsCap = False
            for l in "abcdefghijklmnopqrstuvwxyz"[:n]:
                # check A in w => a in w
                if l.capitalize() in w:
                    containsCap = True
                    if l not in w:
                        use = False
                        break

            if k > 0 and not containsCap:
                use = False
            else:
                v = getLowerNoDublicates(w1 + v2)
                if len(v) > 4:  # TODO otherwise the graph gets too large
                    use = False
                if not v == "abcdefghijklmnopqrstuvwxyz"[:len(v)]:
                    use = False
                else:
                    for i in range(n):
                        if w1[i] == w2[i]:
                            use = False
                            break

            if use:
                phis += [(w1, w2)]

    return phis


def isGraphInteresting(G):
    # contains B2
    # use graph_tool.topology.subgraph_isomorphism

    # contains T3
    return True


def removeStrayEdges(G):
    uG = G.to_undirected(G)
    for v in G.nodes:
        nbs = list(uG.neighbors(v))
        # print(v,nbs)
        if len(nbs) == 1:
            u = nbs[0]
            # print(v,u)
            loop = 0
            if G.has_edge(u, u):
                loop = 1
            if (G.has_edge(u, v) and G.out_degree[u] > 1 + loop) or (G.has_edge(v, u) and G.in_degree[u] > 1 + loop):
                G.remove_node(v)
                return removeStrayEdges(G)
    return G


def hasLoop(G):
    return len([n for n in G.nodes if (n, n) in G.edges]) > 0


def getCore(G, timelimit=0.5):
    G = G.copy()
    if isinstance(G, nx.DiGraph) and hasLoop(G):
        return nx.DiGraph([(0, 0)])
    # G = removeStrayEdges(G)

    #remove components

    if isinstance(G, nx.DiGraph):
        changed = True
        while changed:
            changed = False
            Components = list(nx.weakly_connected_components(G))
            Components.sort(key=lambda l:len(l))
            print(len(Components),len(G.nodes),timelimit)
            for i in range(len(Components)):
                for j in range(i+1,len(Components)):
                    try:
                        exisitsHom = ArcCon.existsHom(G.subgraph(Components[i]),None,G.subgraph(Components[j]),componentwise=False,timelimit=timelimit)
                    except:
                        changed = True
                        timelimit = timelimit * 1.2
                        continue
                    if exisitsHom:
                        # print(f)
                        changed = True
                        # reduce G to im(f)
                        rest = Components[i]
                        G.remove_nodes_from(rest)
                        #timelimit = timelimit / 2
                        break
    changed = True
    while changed:
        # print(str((len(G.nodes), len(G.edges))))
        changed = False
        # arccon on G
        fs = ArcCon.initF(G, G)
        fs = ArcCon.arcCon(G, fs, G)[0]
        for v in G.nodes:
            #print(v,len(G.nodes), timelimit)
            for u in fs[v]:
                if u != v:
                    # print(u,v,changed)
                    # try to contract u and v by finding endo with f(u)=f(v)=u or f(u)=f(v)=v
                    # f = dict()
                    f = copy.deepcopy(fs)

                    f[u] = {u}
                    f[v] = {u}
                    try:
                        f = ArcCon.findHom(G, G, f, timelimit=timelimit,workingSet={u,v})
                    except:
                        changed = True
                        timelimit = timelimit*1.2
                        continue
                    if f is not None:
                        # print(f)
                        changed = True
                        # reduce G to im(f)
                        im = {f[k] for k in f.keys()}
                        rest = {v for v in G.nodes if v not in im}
                        G.remove_nodes_from(rest)
                        timelimit = max(timelimit/2,0.1)
                        break
            if changed:
                break
    return G


#todo verify corectness and compare time with getCore
def getCore2(G, timelimit=float('inf')):
    G = G.copy()
    if isinstance(G, nx.DiGraph) and hasLoop(G):
        return nx.DiGraph([(0, 0)])
    # G = removeStrayEdges(G)
    changed = True
    while changed:
        # print(str((len(G.nodes), len(G.edges))))
        changed = False
        # arccon on G
        fs = ArcCon.initF(G, G)
        fs = ArcCon.arcCon(G, fs, G)[0]
        for v in G.nodes:
            # print(u,v,changed)
            # try to contract u and v by finding endo with f(u)=f(v)=u or f(u)=f(v)=v
            # f = dict()
            f = copy.deepcopy(fs)

            [f[u].remove(v) for u in f]
            f = ArcCon.findHom(G, G, f, timelimit=timelimit)
            if f is not None:
                # print(f)
                changed = True
                # reduce G to im(f)
                im = {f[k] for k in f.keys()}
                rest = {v for v in G.nodes if v not in im}
                G.remove_nodes_from(rest)
                break
    return G


def getCoreC3Conservative(G, zeros={3}, timelimit=float('inf')):
    G = G.copy()
    if hasLoop(G):
        return nx.DiGraph([(0, 0)])
    # G = removeStrayEdges(G)
    changed = True
    while changed:
        # print(str((len(G.nodes), len(G.edges))))
        changed = False
        # arccon on G
        fs = ArcCon.initF(G, G)
        unRel = {k for k in G.nodes if k[0] == k[1] and {k[s] for s in zeros} == {'0'}}
        print(unRel)
        for k in G.nodes:
            if k[0] == k[1] and {k[s] for s in zeros} == {'0'}:
                fs[k] = unRel.copy()

        fs = ArcCon.arcCon(G, fs, G)[0]
        for v in G.nodes:
            for u in fs[v]:
                if u != v:
                    # print(u,v,changed)
                    # try to contract u and v by finding endo with f(u)=f(v)=u or f(u)=f(v)=v
                    # f = dict()
                    f = copy.deepcopy(fs)

                    f[u] = {u}
                    f[v] = {u}
                    f = ArcCon.findHom(G, G, f, timelimit=timelimit)
                    if f is not None:
                        # print(f)
                        changed = True
                        # reduce G to im(f)
                        im = {f[k] for k in f.keys()}
                        rest = {v for v in G.nodes if v not in im}
                        G.remove_nodes_from(rest)
                        break
            if changed:
                break
    return G


def filterFormulasFor(G, fs, H):
    gs = []
    for f in fs:
        PG = pppower(G, f)
        size = (len(PG.nodes), len(PG.edges))
        PG = getCore(PG)
        if nx.is_isomorphic(PG, H):
            drawGraph(PG, (f, size))
            print(f, size)
            gs += (f, size)
    return gs


# Hs is a list of  digraphs
# fs = pptester.filterFormulasForACworks(pptester.T3,pptester.getReasonableFormulas(pptester.T3,2),[pptester.getPath('11011')])
def filterFormulasForACworks(G, fs, Hs):
    gs = []
    for f in fs:
        PG = pppower(G, f)
        size = (len(PG.nodes), len(PG.edges))
        for i, H in enumerate(Hs):
            h1 = ArcCon.initF(H, PG)
            h1 = ArcCon.arcCon(H, h1, PG)[0]
            if set() not in [h1[k] for k in h1.keys()]:
                # print('embeds',f,size)
                h2 = ArcCon.initF(PG, H)
                h2 = ArcCon.arcCon(PG, h2, H)[0]
                if set() not in [h2[k] for k in h2.keys()]:
                    # drawGraph(PG,(f,size))
                    print(i, f, size)
                    gs += (i, f, size)
    return gs  # use AC to check hom equiv (AC works so plain AC is sufficient)


# compute all connected (acyclic and some cyclic) Digraphs with n vertices
def getSomeGraphs(n):  # getAllGraphs
    if n == 2:
        return [nx.DiGraph([(0, 1)])]
    if n == 3:
        return [nx.DiGraph([(0, 1), (1, 2)]), nx.DiGraph([(0, 1), (1, 2), (0, 2)])]
    Gs = getSomeGraphs(n - 1)  # getAllGraphs

    # add vertex n-1
    Hs = []
    for G in Gs:
        # edge possibilities
        G.add_node(n - 1)
        ws = getWords(n - 1, "01n")
        for w in ws:
            if '0' in w or '1' in w:
                H = G.copy()
                for i in range(len(w)):
                    if w[i] == '0':
                        H.add_edge(i, n - 1)
                    if w[i] == '1':
                        H.add_edge(n - 1, i)
                new = True
                for H0 in Hs:
                    if nx.is_isomorphic(H0, H):
                        new = False
                        break
                if new:
                    Hs += [H]
    return Hs


rct = dict()


def getWordsMaxLen(n, alph, maxLen, letter, length):
    if n == 0:
        return ['']
    ws = []
    for a in alph:
        if a == letter and length < maxLen:
            ws += [a + w for w in getWordsMaxLen(n - 1, alph, maxLen, a, length + 1)]
        elif a != letter:
            ws += [a + w for w in getWordsMaxLen(n - 1, alph, maxLen, a, 1)]
    return ws


def getCycles(n, onlyBalanced=True,onlyCores=True):
    cs = []
    for maxLen in range(1, n - 1):
        cs += ['1' * maxLen + '0' + w + '0' for w in getWordsMaxLen(n - 2 - maxLen, '01', maxLen, '0', 1)]
    if onlyBalanced:
        cs = [c for c in cs if 2 * len([i for i in range(len(c)) if c[i] == '0']) == len(c)]
    # cs = ['1'+w+'0' for w in getWords(n-2,'01')]
    print('no cores', len(cs))
    for w in cs:
        c = getCycle(w)
        if len(c.nodes) == n:
            if not onlyCores or len(getCore(c).nodes)== n:
                yield c
    # Cs = [getCore(getCycle(w)) for w in cs]
    # return [c for c in Cs if len(c.nodes)==n]


def getNoSigma2Cycles(n):
    cs = getCycles(n)
    # print('core cycles',len(cs))
    return [c for c in cs if
            not Identities.satisfysIdentity(c, Identities.Sigma2, partition=getLevelsOfBalancedGraph(c))]


def getNoSiggCycles(n):
    cs = getNoSigma2Cycles(n)
    print('noS2', len(cs))
    cs = [c for c in cs if not Identities.satisfysIdentity(c, Identities.Majority)]
    print('noS2,noMaj', len(cs))

    return [c for c in cs if
            not Identities.satisfysIdentity(c, Identities.Majority) and not Identities.satisfysIdentity(c,
                                                                                                        Identities.Sigg3)]


def getRootedCoreTrees(n, d, onlyCores=True):
    # print('start',n,d)
    if n == 1 and d == 0:
        T = nx.DiGraph()
        T.add_node(0)
        return [(T, 0)]
    if d > n + 1 or d == 0:
        return []
    global rct
    if (n, d, onlyCores) in rct:
        return rct[(n, d, onlyCores)]

    Ts = []
    count = 0
    combTime = 0
    testTime = 0
    for i in range(1, n):
        # print('T', i, d - 1)
        for T, t in getRootedCoreTrees(i, d - 1, onlyCores):
            for d2 in range(0, d + 1)[::-1]:
                # print('S', n - 1, d2)
                for S, s in getRootedCoreTrees(n - i, d2, onlyCores):
                    count += 1
                    start = time.time()
                    newT = T.copy()
                    nodesT = list(newT.nodes)
                    newT = nx.relabel_nodes(newT, {nodesT[j]: j for j in range(len(nodesT))})
                    S = S.copy()
                    nodesS = list(S.nodes)
                    S = nx.relabel_nodes(S, {nodesS[j]: j + len(nodesT) for j in range(len(nodesS))})
                    newT.add_edges_from(S.edges)
                    newt = nodesS.index(s) + len(nodesT)
                    newT1 = newT.copy()
                    newT1.add_edge(nodesT.index(t), newt)
                    newT2 = newT.copy()
                    newT2.add_edge(newt, nodesT.index(t))
                    combTime += time.time() - start
                    start = time.time()
                    if addCoreToTreeList(Ts, newT1, newt, d2 == d, onlyCores):
                        Ts += [(newT1, newt)]
                    if addCoreToTreeList(Ts, newT2, newt, d2 == d, onlyCores):
                        Ts += [(newT2, newt)]
                    testTime += time.time() - start
    print(n, d, 'iterations:', count, 'combine time:', int(combTime * 1000), 'ms test time:', int(testTime * 1000),
          'ms')
    rct[(n, d, onlyCores)] = copy.deepcopy(Ts)
    return Ts


def addCoreToTreeList(Ts, newT, newt, homeqpossible=True, onlyCores=True):
    if not onlyCores:
        return True
    if ArcCon.isTreeCore(newT, {newt: {newt}}, workingSet={newt}):
        if homeqpossible:
            for (T, t) in Ts:
                #                if nx.is_isomorphic(newT,T,{newt:t}):
                if ArcCon.existsHom(T, {t: {newt}}, newT, ACWorks=True, componentwise=False,
                                    workingSet={t}) and ArcCon.existsHom(newT, {newt: {t}}, T, ACWorks=True,
                                                                         componentwise=False, workingSet={newt}):
                    # print('hom eq')
                    return False
        return True
    return False


# no double edges
def getRandomGraph(n, edgepropability=0.1, minEdges=1, maxEdges=1000):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = random.random()
            if edges < maxEdges and r < edgepropability / 2:
                G.add_edge(i, j)
                edges += 1
            elif edges < maxEdges and r < edgepropability:
                G.add_edge(j, i)
                edges += 1
    if len(G.edges) < minEdges:
        return getRandomGraph(n, edgepropability, minEdges, maxEdges)
    return G


def getRandomTSCoreWithHM2(n, edgepropability=0.1, minEdges=1, maxEdges=1000):
    while True:
        G = getRandomGraph(n, edgepropability, minEdges, maxEdges)
        if min([G.degree[v] for v in G.nodes]) == 0:
            continue
        if not Identities.satisfysIdentity(G, Identities.HM2):
            continue
        if Identities.satisfysIdentity(G, Identities.Const):
            continue
        G = getCore(G)
        if len(G.nodes) < n:
            continue
        if not isTotallySymmetric(G):
            continue
        return G


# return all Digraphs with n vertices up to isomorphism also disconnected double edges
def getReallyAllGraphs(n, doubleEdges=True, maxEdges=1000):
    if n == 2:
        if doubleEdges:
            return [nx.DiGraph([(0, 1)]), nx.DiGraph([(0, 1), (1, 0)])]
        else:
            return [nx.DiGraph([(0, 1)])]
    # if n == 3:
    #    return [nx.DiGraph([(0, 1), (1, 2)]), nx.DiGraph([(0, 1), (1, 2), (0, 2)])]
    Gs = getReallyAllGraphs(n - 1, doubleEdges, maxEdges)

    # add vertex n-1
    Hs = []
    for G in Gs:
        # edge possibilities
        G.add_node(n - 1)
        edges = len(G.edges)
        alph = '01n'
        if doubleEdges:
            alph += '2'
        ws = getWords(n - 1, alph)
        for w in ws:
            H = G.copy()
            for i in range(len(w)):
                if (w[i] == '0' or w[i] == '2') and edges < maxEdges:
                    H.add_edge(i, n - 1)
                if (w[i] == '1' or w[i] == '2') and edges < maxEdges:
                    H.add_edge(n - 1, i)
            new = True
            for H0 in Hs:
                if nx.is_isomorphic(H0, H):
                    new = False
                    break
            if new:
                Hs += [H]
    return Hs


def getPath(w):
    n = len(w)
    return nx.DiGraph(
        [(i, (i + 1)) for i in range(n) if w[i] == '1'] + [((i + 1), i) for i in range(n) if w[i] == '0'])


def getCycle(w):
    n = len(w)
    return nx.DiGraph(
        [(i, (i + 1) % n) for i in range(n) if w[i] == '1'] + [((i + 1) % n, i) for i in range(n) if w[i] == '0'])


def getTwoBraidedPaths(n):
    ps = getRootedTwoBraidedPaths(n)
    return [p for p in ps if ArcCon.isTreeCore(p)]


def getRootedTwoBraidedPaths(n):
    if n == 1:
        point = nx.DiGraph()
        point.add_node(0)
        return [point]
    ps = getRootedTwoBraidedPaths(n - 1)
    res = []
    for p in ps:
        q = p.copy()
        p.add_edge(n - 2, n - 1)
        q.add_edge(n - 1, n - 2)
        if ArcCon.isTreeCore(p, {0: {0}}):
            res += [p]
        if ArcCon.isTreeCore(q, {0: {0}}):
            res += [q]
    return res


def getPathsUpTo(n):
    pss = [getRootedPaths(k) for k in range(2, n + 1)]
    res = []
    for ps in pss:
        for p in ps:
            if ArcCon.isTreeCore(p):
                new = True
                for t in res:
                    if nx.is_isomorphic(t, p):  # ArcCon.isHomEqACworks(t,p):
                        new = False
                        break
                if new:
                    res += [p]
    return res

    # return [p for ps in pss for p in ps if ArcCon.isTreeCore(p)]


def getCorePaths(n):
    combineTime = 0
    start = time.time()
    if n % 2 == 0:
        cps = getRootedPaths(n // 2)
        print('numberOfRootedCorePaths', len(cps))
        # ps = []
        for i in range(len(cps)):
            for j in range(len(cps)):
                startComb = time.time()
                p = cps[i].copy()
                p2 = nx.relabel_nodes(cps[j], {v: (v + n // 2) for v in cps[j].nodes})
                p.add_edges_from(p2.edges)
                p.add_edge(n // 2 - 1, n - 1)
                combineTime += time.time() - startComb
                # drawGraph(p)

                if ArcCon.isTreeCore(p, workingSet={n - 1}):
                    yield p
    else:
        cps = getRootedPaths((n + 1) // 2)
        print('numberOfRootedCorePaths', len(cps))
        # ps = []
        for i in range(len(cps)):
            for j in range(i + 1, len(cps)):
                startComb = time.time()
                p = cps[i].copy()
                p2 = nx.relabel_nodes(cps[j], {v: (v + (n + 1) // 2) for v in cps[j].nodes})
                p.add_edges_from(p2.edges)
                p = nx.contracted_nodes(p, (n + 1) // 2 - 1, n, False)
                combineTime += time.time() - startComb
                # drawGraph(p)

                if ArcCon.isTreeCore(p, workingSet={n - 1}):
                    yield p
    print('combTime', combineTime)
    print('allTime', time.time() - start)
    # return ps


def storeCorePathsToFile(n):
    ps = getCorePaths(n)
    f = open(r'pathsList\\' + str(n), 'w+')
    for p in ps:
        f.writelines([pathToNumbers(p) + '\n'])
    f.close()


def getCorePathsFromFile(n):
    ls = open(r'pathsList\\' + str(n)).readlines()
    for l in ls:
        p = getPath(l)
        yield p

def getBalancedPaths():
    l = 2
    while True:
        for onePos in itertools.combinations(range(0,l),l//2):
            w = ''
            for i in range(0,l):
                w += '1' if i in onePos else '0'
            yield w
        l += 2

def testHMBound(n):
    ls = open(r'pathsHM\\' + str(n)).readlines()
    for l in ls:
        [p,HM,flip] = l.split(', ')
        if int(HM)<30 and int(HM)+2 > len(p.split(' ')):
            print(p,',',HM)

def canPathPPdefineOrd(path):
    ws = getBalancedPaths()
    lvl = getLevelsOfBalancedGraph(path)
    pathMaxNumber = max([int(a) for a in zopathToNumberpath(pathToNumbers(path),',').split(',')[:-1]])

    for w in ws:
        if not '0'*(pathMaxNumber+1) in w and not '1'*(pathMaxNumber+1) in w:
            #if len(w)>21:
            #    print(w,pathMaxNumber)
            p= getPath(w)
            f = ArcCon.arcCon(p,None,path)[0]
            imageStartpoint = f[0]
            imageEndpoint = f[len(w)]
            intersection = imageStartpoint.intersection(imageEndpoint)
            for a in intersection:
                for b in intersection:
                    aLvL = [k for k in lvl if a in lvl[k]][0]
                    bLvL = [k for k in lvl if b in lvl[k]][0]
                    if a < b and aLvL==bLvL:

                        newf= f.copy()
                        newf[0] = {a}
                        newf[len(w)] = {a}
                        aToa = ArcConFast.existsHom(p,newf,path,ACWorks= True)
                        if aToa:
                            newf= f.copy()
                            newf[0] = {b}
                            newf[len(w)] = {b}
                            bTob = ArcConFast.existsHom(p,newf,path,ACWorks= True)
                            if bTob:
                                newf= f.copy()
                                newf[0] = {a}
                                newf[len(w)] = {b}
                                aTob = ArcConFast.existsHom(p,newf,path,ACWorks= True)
                                if aTob:
                                    newf = f.copy()
                                    newf[0] = {b}
                                    newf[len(w)] = {a}
                                    bToa = ArcConFast.existsHom(p,newf,path,ACWorks= True)
                                    if not bToa:
                                        aLvL = [k for k in lvl if a in lvl[k]][0]
                                        wLong = w*(len(lvl[aLvL])-1)
                                        #print((len(lvl[aLvL])-1))
                                        if not ArcCon.existsHom(getPath(wLong),{0:{b},len(wLong):{a}},path,ACWorks=True):
                                            #todo test w with small levenstein distance to path first
                                            print('path',zopathToNumberpath(pathToNumbers(path)),'Formula ',zopathToNumberpath(w),w,a,b)
                                            return (w,a,b)


def getNLHardCorePathsfromFile(n,printPaths=False):
    ls = open(r'pathsHM\\' + str(n)).readlines()
    for l in ls:
        [p,HM,flip] = l.split(', ')
        if int(HM) == 30:
            if printPaths:
                print(p,flip)
            p = [int(a) for a in p.split(' ')]
            w = ''
            edge = '1'
            for a in p:
                w += edge*a
                edge = '1' if edge == '0' else '0'
            p = getPath(w)
            yield p


def getRootedPaths(n):
    if n == 1:
        point = nx.DiGraph()
        point.add_node(0)
        return [point]
    ps = getRootedPaths(n - 1)
    res = []
    for p in ps:
        q = p.copy()
        p.add_edge(n - 2, n - 1)
        q.add_edge(n - 1, n - 2)
        if ArcCon.isTreeCore(p, {n - 1: {n - 1}}, {n - 1}):
            res += [p]
        if ArcCon.isTreeCore(q, {n - 1: {n - 1}}, {n - 1}):
            res += [q]
    return res


# todo solve for short lists first
# return all solutions s:x ->H such that s can be extended to a homomorphism from G to H
def getAllSolutions(G, x, f, H, ACWorks=False, first=True, timelimit=float('inf'), workingSet=None):
    global start
    # print(time.time() - start,timelimit,(time.time() - start) > timelimit)
    if (time.time() - start) > timelimit:
        raise Exception('timeout')

    if len(x) == 0:
        if ArcCon.existsHom(G, f, H, ACWorks, workingSet=workingSet):
            return {()}
        return set()
    if first:
        f = ArcCon.initF(G, H, f)
    f = ArcCon.arcCon(G, f, H, workingSet=workingSet)[0]
    res = set()
    # print(G.nodes,f)
    for v in f[x[0]]:
        ff = copy.deepcopy(f)
        ff[x[0]] = {v}
        res = res.union({(v,) + sol for sol in
                         getAllSolutions(G, x[1:], ff, H, ACWorks, False, timelimit=timelimit, workingSet={x[0]})})
    if len(res) > 2000:
        print('sols', len(res), 'varsLeft', len(x))
    return res


def getAllSolutionsFast(G, x, f, H, ACWorks=False, first=True, timelimit=float('inf'), workingSet=None, todoTuple=None):
    global start
    # print(time.time() - start,timelimit,(time.time() - start) > timelimit)
    if (time.time() - start) > timelimit:
        raise Exception('timeout')

    if len(x) == 0:
        if ArcCon.existsHom(G, f, H, ACWorks, workingSet=workingSet):
            return [()]
        return []
    if first:
        f = ArcCon.initF(G, H, f)
        x = [(a, i) for i, a in enumerate(x)]
        todoTuple = ()
    f = ArcCon.arcCon(G, f, H, workingSet=workingSet)[0]
    res = []
    # print(G.nodes,f)
    x = sorted(x, key=lambda a: len(f[a[0]]))
    if first:
        print([len(f[a[0]]) for a in x])
    for i, v in enumerate(f[x[0][0]]):
        ff = copy.deepcopy(f)
        ff[x[0][0]] = {v}
        res += [((v, x[0][1]),) + sol for sol in
                getAllSolutionsFast(G, x[1:], ff, H, ACWorks, False, timelimit=timelimit, workingSet={x[0][0]},
                                    todoTuple=todoTuple + (len(f[x[0][0]]) - i - 1,))]

        # percent += 1/(len(f[x[0][0]])*reduce(lambda a,b:(b+1)*a, todoTuple, 1))
    if len(res) > 2000:
        print('sols', len(res), todoTuple)
    if first:
        res = [tuple([a[0] for a in sorted(sol, key=lambda a: a[1])]) for sol in res]
    return res


# not actualla faster but yield is hopefully better for memory
def getAllSolutionsVeryFast(G, x, f, H, ACWorks=False, first=True, timelimit=float('inf'), workingSet=None,
                            todoTuple=None, sol=()):
    global start
    # print(time.time() - start,timelimit,(time.time() - start) > timelimit)
    if (time.time() - start) > timelimit:
        raise Exception('timeout')

    if len(x) == 0:
        if ArcCon.existsHom(G, f, H, ACWorks, workingSet=workingSet):
            yield tuple([a[0] for a in sorted(sol, key=lambda a: a[1])])
        return
    if first:
        f = ArcCon.initF(G, H, f)
        x = [(a, i) for i, a in enumerate(x)]
        todoTuple = ()
    f = ArcCon.arcCon(G, f, H, workingSet=workingSet)[0]
    #res = []
    # print(G.nodes,f)
    x = sorted(x, key=lambda a: len(f[a[0]]))
    #if first:
    #    print([len(f[a[0]]) for a in x])
    numberOfSols = 0
    for i, v in enumerate(f[x[0][0]]):
        ff = copy.deepcopy(f)
        ff[x[0][0]] = {v}
        for solution in getAllSolutionsVeryFast(G, x[1:], ff, H, ACWorks, False, timelimit=timelimit, workingSet={x[0][0]},
                                    todoTuple=todoTuple + (len(f[x[0][0]]) - i - 1,),sol=sol+((v,x[0][1]),)):
            yield solution
            numberOfSols +=1

        # percent += 1/(len(f[x[0][0]])*reduce(lambda a,b:(b+1)*a, todoTuple, 1))
    if numberOfSols>2000: print('sols',numberOfSols, todoTuple)
    #if first:
    #    res = [tuple([a[0] for a in sorted(sol, key=lambda a: a[1])]) for sol in res]
    #return res
dynSols=dict()

#dynamic programming works because we have only binary constraints
#much faster than getAllSolutionsFast (at least 10 times)
def getAllSolutionsDyn(G, x, f, H, ACWorks=False, first=True, timelimit=float('inf'), workingSet=None, todoTuple=None):
    global start,dynSols
    #print(dynSols.keys())
    # print(time.time() - start,timelimit,(time.time() - start) > timelimit)
    if (time.time() - start) > timelimit:
        raise Exception('timeout')

    if len(x) == 0:
        if ArcCon.existsHom(G, f, H, ACWorks, workingSet=workingSet):
            return [()]
        return []
    if first:
        dynSols = dict()
        f = ArcCon.initF(G, H, f)
        x = [(a, i) for i, a in enumerate(x)]
        todoTuple = ()
    f = ArcCon.arcCon(G, f, H, workingSet=workingSet)[0]
    res = []
    # print(G.nodes,f)
    dynKey= None
    if min([len(f[a[0]]) for a in x])>1:
        dynKey = frozenset([(frozenset(f[a[0]]),a[1]) for a in x])
    #print(dynKey)
    if dynKey in dynSols:
        #print('used dyn')
        return dynSols[dynKey]

    x = sorted(x, key=lambda a: len(f[a[0]]))
    for i, v in enumerate(f[x[0][0]]):
        ff = copy.deepcopy(f)
        ff[x[0][0]] = {v}
        res += [((v, x[0][1]),) + sol for sol in
                getAllSolutionsDyn(G, x[1:], ff, H, ACWorks, False, timelimit=timelimit, workingSet={x[0][0]},
                                    todoTuple=todoTuple + (len(f[x[0][0]]) - i - 1,))]

        # percent += 1/(len(f[x[0][0]])*reduce(lambda a,b:(b+1)*a, todoTuple, 1))
    if len(res) > 2000:
        print('sols', len(res), todoTuple)
    if first:
        if len(res) > 2000: print('number of dynkeys ',len(dynSols.keys()))
        res = [tuple([a[0] for a in sorted(sol, key=lambda a: a[1])]) for sol in res]
    else:
        if dynKey is not None: dynSols[dynKey]=res
    return res

# return graph H such that H has an edge from u to v iff x -> u, y -> v can be extended to a homomorphism Gadget -> G
# x and y are tuples of nodes in Gadget of the same length
def pppowerWithGraph(G, Gadget, x, y, ACWorks=False, f=None, timelimit=float('inf'), noExistentialFirst=False):
    # if not isinstance(Gadget, nx.DiGraph):
    #    Hs = [pppowerWithGraph(G,Gad,x,y,ACWorks,f) for Gad in Gadget.Gs]
    #    return Structures.Structure(Hs)
    # else:
    global start
    start = time.time()
    if noExistentialFirst:
        GadgetNoEx = copy.deepcopy(Gadget)
        GadgetNoEx.remove_nodes_from(set(Gadget.nodes).difference(set(x + y)))
        # print(GadgetNoEx.edges())
        edgesNoEx = getAllSolutions(GadgetNoEx, x + y, f, G, ACWorks, timelimit=timelimit)
        print('no existential quantifiers', len(edgesNoEx))
        edges = []
        for e in edgesNoEx:
            print(len(edges))
            if ArcCon.findHom(Gadget, G, {(x + y)[i]: {e[i]} for i in range(len(e))}, ACWorks):
                edges += [e]

    else:
        # edges = getAllSolutions(Gadget, x + y, f, G, ACWorks,timelimit=timelimit)
        #s = time.time()
        #edgesSlow = getAllSolutions(Gadget, x + y, f, G, ACWorks, timelimit=timelimit)
        #print('slow', len(edgesSlow), time.time() - s)
        #s = time.time()
        #edges = getAllSolutionsFast(Gadget, x + y, f, G, ACWorks, timelimit=timelimit)
        #print('fast', len(edges), time.time() - s,set(edges).symmetric_difference(set(edgesSlow)))
        #s=time.time()
        #edgesFast = getAllSolutionsVeryFast(Gadget, x + y, f, G, ACWorks, timelimit=timelimit)
        #edgesFast=list(edgesFast)
        #print('veryfast', len(edgesFast), time.time() - s)
        #s = time.time()
        if set(Gadget.nodes) == set(x + y):
            edges = getAllSolutionsDyn(Gadget, x + y, f, G, ACWorks, timelimit=timelimit)
        else:
            edges = getAllSolutionsFast(Gadget, x + y, f, G, ACWorks, timelimit=timelimit)

        #print('fast', len(edges), time.time() - s,set(edges).symmetric_difference(set(edgesFast)))

        # print(len(edges),len(edgesFast),edges.symmetric_difference(edgesFast))
    # print(time.time() - start,timelimit)
    # print(edges)
    H = nx.DiGraph()
    n = len(x)
    for e in edges:
        H.add_edge(str(e[:n]), str(e[n:]))
    #print('size of graph',sys.getsizeof(H.edges)+sys.getsizeof(H.nodes))
    return H


def ppdefWithGraph(G, Gadget, x, y, ACWorks=False):
    H = nx.DiGraph()
    for u in G.nodes:
        for v in G.nodes:
            f = {x: {u}, y: {v}}
            if ArcCon.findHom(Gadget, G, f, ACWorks):
                H.add_edge(u, v)
    return H


def getSubSets(A: set, k):
    if k == 0:
        return {frozenset()}
    ss = getSubSets(A, k - 1)
    res = set()
    for s in ss:
        for a in A.difference(s):
            res.add(frozenset(s.union({a})))
    return res


# input u=(u1,u2,u3)
# returns u or (ui,ui,ui) if ui appears twice
def majority(u):
    if u[0] == u[1]:
        return (u[1], u[1], u[1])
    if u[0] == u[2]:
        return (u[0], u[0], u[0])
    if u[1] == u[2]:
        return (u[1], u[1], u[1])
    return u


def hasMajority(G, ACWorks=False):
    MG = nx.DiGraph()
    for e1 in G.edges:
        for e2 in G.edges:
            for e3 in G.edges:
                u = (e1[0], e2[0], e3[0])
                v = (e1[1], e2[1], e3[1])
                u = majority(u)
                v = majority(v)

                MG.add_edge(u, v)
    f = {(u, u, u): {u} for u in G.nodes}
    return ArcCon.findHom(MG, G, f, ACWorks)


def PowerSetGraphHasEdge(G, u, v):
    for Gu in u:
        if len(set(G.successors(Gu)).intersection(v)) == 0:
            return False
    for Gv in v:
        if len(set(G.predecessors(Gv)).intersection(u)) == 0:
            return False
    return True


def getPowerSetGraphSlow(G, n=None, relabel=True):
    if n is None:
        n = len(G.nodes)
    PG = nx.DiGraph()

    sources = {v for v in G.nodes if G.out_degree[v] == 0}
    sinks = {v for v in G.nodes if G.in_degree[v] == 0}
    # add nodes
    for k in range(1, n + 1):
        for S in getSubSets(set(G.nodes), k):
            # PG.add_nodes_from(getSubSets(set(G.nodes), k))
            if len(S.intersection(sinks)) == 0 or len(S.intersection(sources)) == 0:
                PG.add_node(S)
    # add edges
    # print('start',PG.nodes)
    for u in PG.nodes:
        v = set()
        for Gu in u:
            v = v.union(set(G.successors(Gu)))
        workingLs = [frozenset(v)]
        while len(workingLs) > 0:
            v = workingLs.pop()
            if PowerSetGraphHasEdge(G, u, v):
                if len(v) <= n:
                    PG.add_edge(u, v)
                for Gv in v:
                    workingLs += [v.difference({Gv})]

    if relabel:
        return nx.relabel_nodes(PG, lambda v: str(set(v)))
    return PG


def getPowerSetGraph(G, n=None, relabel=True):
    if n is None:
        n = len(G.nodes)
    PG = nx.DiGraph()

    sources = {v for v in G.nodes if G.out_degree[v] == 0}
    sinks = {v for v in G.nodes if G.in_degree[v] == 0}
    smooth = {v for v in G.nodes if G.in_degree[v] != 0 and G.out_degree[v] != 0}
    # add nodes
    for k in range(1, n + 1):
        for ks in range(0, k + 1):
            for Ssmooth in getSubSets(smooth, k - ks):
                for Ssinks in getSubSets(sinks, ks):
                    PG.add_node(Ssmooth.union(Ssinks))
                for Ssources in getSubSets(sources, ks):
                    PG.add_node(Ssmooth.union(Ssources))

    # add edges
    # print('start',PG.nodes)
    for u in PG.nodes:
        v = set()
        for Gu in u:
            v = v.union(set(G.successors(Gu)))
        workingLs = [frozenset(v)]
        while len(workingLs) > 0:
            v = workingLs.pop()
            if PowerSetGraphHasEdge(G, u, v):
                if len(v) <= n:
                    PG.add_edge(u, v)
                for Gv in v:
                    workingLs += [v.difference({Gv})]

    if relabel:
        return nx.relabel_nodes(PG, lambda v: str(set(v)))
    return PG


# [ArcConFast.existsHom(pptester.ManuelsPowersetGraph(G,2),{tuple([frozenset([v])]*2):{v} for v in G.nodes},G,timelimit=20) for G in Gs]
def ManuelsPowersetGraph(G, n):
    PG = getPowerSetGraph(G, relabel=False)
    IG = nx.DiGraph()
    for es in itertools.product(PG.edges, repeat=n):
        u = tuple([e[0] for e in es])
        v = tuple([e[1] for e in es])
        # print(es,u,[len(s) for s in u],v,[len(s) for s in v])
        if 1 in [len(s) for s in u] and 1 in [len(s) for s in v]:
            IG.add_edges_from([(u, v)])
    return IG


def getPPConstructableGraphs(G, cs, Gs=[], ACWorks=False, coreify=True):
    for c in cs:
        if c.isApplicable(G):
            H = c.apply(G)
            if coreify:
                H = getCore(H)
                new = True
                for (K, _) in Gs:
                    if nx.is_isomorphic(H, K):
                        new = False
                        break
                if new:
                    Gs += [(H, c)]
    return Gs


def testgetIncomparableSubsets():
    sets = getSubSets({1, 2, 3}, 1)
    sets.update(getSubSets({1, 2, 3}, 2))
    sets.update(getSubSets({1, 2, 3}, 3))
    s = getIncomparableSubsets(sets)
    return s


def getIncomparableSubsets(sets):
    print('sets', [set(s) for s in sets])
    res = [frozenset()]
    sets2 = sets.copy()
    for S in sets:
        print(S)
        sets2.remove(S)
        subsets = getIncomparableSubsets({R for R in sets2 if not R.issubset(S) and not R.issuperset(S)})
        print(subsets)
        res += [frozenset({S}.union(T)) for T in subsets]
    return res


def getkLayeredABSGraphTreeSameLvlComponent(G, k, relabel=True):
    BlockSymG = nx.DiGraph()
    ACG = getPowerSetGraphTreeSameLvlComponent(G, relabel=False)  # use for edges
    lvl = getLevelsOfBalancedGraph(G)
    # print(lvl)
    for l in lvl:
        sets = set()
        for k in range(1, len(G.nodes) + 1):
            sets.update(getSubSets(lvl[l], k))
        [BlockSymG.add_nodes_from(getSubSets(sets, m)) for m in range(1, len(sets) + 1)]
    print(len(BlockSymG.nodes), 65790)
    for vs in BlockSymG.nodes:
        print(vs)
        for ws in BlockSymG.nodes:
            hasEdge = True
            for v in vs:
                if len(set(ACG.successors(v)).intersection(ws)) == 0:
                    hasEdge = False
                    break
            if hasEdge:
                for w in ws:
                    if len(set(ACG.predecessors(w)).intersection(vs)) == 0:
                        hasEdge = False
                        break
            if hasEdge:
                BlockSymG.add_edge(vs, ws)
    return BlockSymG

    # compute all outgoing edges
    # for v in vs:
    #    ACG.successors(v)

    # print(Sss)
    # PG.add_nodes_from([s for s in getIncomparableSubsets(sets) if len(s)>0])


def getPowerSetGraphTreeSameLvlComponent(G, relabel=True):
    PG = nx.DiGraph()

    lvl = getLevelsOfBalancedGraph(G)
    for k in range(1, len(G.nodes) + 1):
        for l in lvl:
            PG.add_nodes_from(getSubSets(lvl[l], k))

    for u in PG.nodes:
        v = set()
        for Gu in u:
            v = v.union(set(G.successors(Gu)))
        workingLs = [frozenset(v)]
        while len(workingLs) > 0:
            v = workingLs.pop()
            if PowerSetGraphHasEdge(G, u, v):
                PG.add_edge(u, v)
                for Gv in v:
                    workingLs += [v.difference({Gv})]

    if relabel:
        return nx.relabel_nodes(PG, lambda v: str(set(v)))
    return PG


def isTotallySymmetric(G, n=None, skip=False, printres=False, isTree=False):
    #    print(2,G.edges)
    if len(G.nodes) <= 3:
        return Identities.satisfysIdentity(G, Identities.TS3)
    if n == None:
        n = len(G.nodes)
    if not skip:
        PG = getPowerSetGraph(G, 2)
        if not ArcCon.existsHom(PG, {str({v}): {v} for v in G.nodes}, G):
            return False
        #    print(3)
        PG = getPowerSetGraph(G, 3)
        if not ArcCon.existsHom(PG, {str({v}): {v} for v in G.nodes}, G):
            return False

    #    print(4)
    if isTree:
        PG = getPowerSetGraphTreeSameLvlComponent(G)
    else:
        PG = getPowerSetGraph(G)
    if ArcCon.existsHom(PG, {str({v}): {v} for v in G.nodes}, G):
        if printres:
            print('\033[92m', True, '\033[0m')
        return True
    if printres:
        print('\033[91m', False, '\033[0m')
    return False


# gives really all graphs
def getAllGraphs(n):
    ws = itertools.product([True, False], repeat=n * n)
    Gs = []
    for w in ws:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(n)))
        for i in range(n):
            for j in range(n):
                if w[i * n + j]:
                    G.add_edge(i, j)
        Gs += [G]
    return Gs


# no loops, always 0 -> 1 -> 2
def getMostGraphs(n):
    ws = itertools.product([True, False], repeat=n * n - n - 2)
    Gs = [nx.DiGraph([(0, 0)]), nx.DiGraph([(0, 1)])]
    for w in ws:
        G = nx.DiGraph([(0, 1), (1, 2)])
        G.add_node(0)
        for i in range(n):
            for j in range(0, i):
                if w[i * n + j - 2 - i]:
                    G.add_edge(i, j)

            for j in range(i + 1, n):
                if w[i * n + j - 2 - i - 1]:
                    G.add_edge(i, j)
        Gs += [G]
    return Gs


# G = pppower(H,("0a","A0"))

# print(getReasonableFormulas(T,2))


H = nx.DiGraph()
H.add_edges_from([(0, 1), (1, 0), (1, 1)])

T3 = nx.DiGraph()
T3.add_edges_from([(0, 1), (1, 2), (0, 2)])

Triad = nx.DiGraph()
Triad.add_edges_from([(0, 1), (2, 1), (3, 2), (4, 3), (7, 0), (0, 5), (5, 6)])

T3C2 = nx.DiGraph()
T3C2.add_edges_from([(3, 4), (4, 3)])  # C2
T3C2.add_edges_from([(0, 1), (1, 2), (0, 2)])  # T3

B2C2 = nx.DiGraph()
B2C2.add_edges_from([(3, 2), (2, 3)])  # C2
B2C2.add_edges_from([(0, 1), (1, 0), (1, 1)])  # B2

C5 = nx.DiGraph()
C5.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

Zig = nx.DiGraph()
Zig.add_edges_from([(0, 1), (1, 2), (3, 2), (3, 4), (4, 5)])

Zig2 = nx.DiGraph([(0, 1), (1, 2), (3, 2), (3, 4), (5, 4), (5, 6), (6, 7)])

Zig3 = nx.DiGraph([(0, 1), (1, 2), (3, 2), (3, 4), (5, 4), (5, 6), (7, 6), (7, 8), (8, 9)])

Zig4 = nx.DiGraph([(0, 1), (1, 2), (3, 2), (3, 4), (5, 4), (5, 6), (7, 6), (7, 8), (9, 8), (9, 10), (10, 11)])

ZigCyc = nx.DiGraph(
    [(0, 1), (1, 2), (3, 2), (3, 4), (4, 5), (5, 6), (7, 6), (8, 7), (8, 9), (10, 9), (11, 10), (0, 11)])

Ord = nx.DiGraph()
Ord.add_edges_from([(0, 0), (0, 1), (1, 1)])

N5 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)])

M5 = nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3), (0, 4)])

M7 = nx.DiGraph([(0, 1), (2, 1), (2, 3), (4, 3), (4, 5), (6, 5), (0, 6)])

N7 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 4)])

T4 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)])

T4b = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)])

T4c = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)])

T5 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 3), (1, 4), (0, 4)])

T5b = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 3), (1, 4)])

T6 = nx.DiGraph(
    [(i, j) for i in range(6) for j in range(6) if i < j])

ThreeP = nx.DiGraph([(0, 1), (0, 2), (2, 1), (0, 3), (3, 4), (4, 1)])

T311 = nx.DiGraph([(0, 1), (1, 2), (0, 2), (2, 3), (4, 0)])

HM1 = nx.DiGraph([(0, 1), (2, 1), (2, 3)])

HM2 = nx.DiGraph([(0, 1), (1, 2), (3, 4), (4, 5), (3, 1), (4, 2)])

T4xC2 = nx.DiGraph(
    [(i + 'a', j + 'b') for i in '0123' for j in '0123' if int(j) > int(i)] + [(i + 'b', j + 'a') for i in '0123' for j
                                                                               in '0123' if int(j) > int(i)])

T4xC3 = nx.DiGraph([(i + 'a', j + 'b') for i in '0123' for j in '0123' if int(j) > int(i)])
T4xC3.add_edges_from([(i + 'b', j + 'c') for i in '0123' for j in '0123' if int(j) > int(i)])
T4xC3.add_edges_from([(i + 'c', j + 'a') for i in '0123' for j in '0123' if int(j) > int(i)])

Crown3 = getCycle('101010')

OrdDi = nx.DiGraph(
    [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)])  # ('aabC0', 'cBC1a'),('abcdAC', 'D1abb0'), ('abAC0', 'BCc1A')
OrdDiSmall = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)])


# Ord <= T3+T3 ('abcc0', '1cCAB')

def analyseC2nHMPolys(n):
    G=getnGk(0,n, getCycle('11'))
    H, q = Identities.getIndicatorGraph(G, Identities.getHM(2*n+1), True)
    f = ArcConFast.findHom(H, G)
    tuples=[(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    for i in range(n+2):
        for j in range(n+2):
            if i != j:
                tuples += [(i,j,j)]
    print(tuples)
    for label in 'abcdefghijklmnopqrstuvw'[:2*n+1]:
        print([f[q[(label,) + t]] for t in tuples])

def getnGk(n,k,G:nx.DiGraph):
    G=G.copy()
    iLabel=1
    jLabel=-1
    for i in range(k):
        while iLabel in G.nodes:
            iLabel += 1
        G.add_edges_from([(v, iLabel) for v in G.nodes])
    for i in range(n):
        while jLabel in G.nodes:
            jLabel -= 1
        G.add_edges_from([(jLabel,v) for v in G.nodes])
    return G


def getT3plusN(n=1,addAtTop=True):
    G = T3.copy()
    G.add_edges_from([(i, i + 1) for i in range(3, n + 2)])
    if addAtTop:
        G.add_edge(2,3)
    else:
        G.add_edge(0,3)
    return G


def getTnplusK(n=1, k=0):
    G = nx.DiGraph([(i, j) for i in range(n - 1) for j in range(i + 1, n)])
    G.add_edges_from([(i, i + 1) for i in range(n - 1, n + k - 1)])
    return G


# Ord <= OrdDi
# 0 ('abcBD0', 'DADd1B') (32, 54)
# 0 ('aabcCD', 'BdDa01') (33, 54)
# 0 ('aabccD', 'BdDa01') (25, 36)
# 0 ('aabAC0', 'cBCB1a') (21, 27)

NLPath = getPath('1111001001111')

GadgetNLPath = NLPath.copy()
GadgetNLPath.add_edges_from(
    [(9, 14), (14, 15), (16, 15), (16, 17), (17, 18), (19, 18), (20, 19), (21, 20), (22, 18), (23, 22), (23, 24),
     (25, 24), (26, 25)])

C3C3 = getCycle('111')
C3C3.add_edges_from([(3, 4), (4, 5), (5, 3)])
C3C3.add_edges_from([(u, v) for u in [0, 1, 2] for v in [3, 4, 5]])

# G = getCore(nx.DiGraph([(0,1),(1,0),(2,3),(3,4),(4,5),(5,2)]))
# drawGraph(G)


# G = pppower(Zig,("ab","AB"))
# drawGraph(G)

# print(getOrderedWords("ABCDEFGHIJKLMNOPQRSTUVWXYZ",4,1))
# print(getOrderedWords("ABC",3,5))
# print(getFormulaFirstTuple("01",3))

# N7.add_edges_from([(7,0),(4,8)])
# N5.add_edges_from([(5,0),(3,6)])
# N5.add_edges_from([(5,6),(6,7),(5,7)])
# T4.add_edges_from([(3,4)])
# T5.remove_edges_from([(0,4),(1,3),(0,3),(1,4)])
# T5.remove_edges_from([(0,4),(1,4)])


# G=pppowerB3(Ord,('aa','c0',[(0,1,2),(0,2,3)])) # T3
# G=pppowerB3(Ord,('abac','def0',[(0,1,3),(1,2,4),(2,3,5),(3,0,6)]))
# drawGraph(G)
# G=getCore(G)
# drawGraph(G)
# Gs = getAllGraphs(3)
# for G in Gs:
#    drawGraph(G)
#    time.sleep(0.3)
PPG = getCore(pppower(Ord, ('abc0', '1AbC')))
PPG = nx.relabel_nodes(PPG, dict([(b, i) for (i, b) in enumerate(PPG.nodes)]))

NoSigma2ButSiggersTree = nx.DiGraph(
    [(0, 1), (2, 1), (3, 2), (4, 3), (5, 0), (5, 6), (6, 7), (7, 8), (0, 9), (10, 9), (11, 10), (10, 12), (12, 13),
     (14, 0), (14, 15), (18, 17), (17, 15), (15, 16)])
NPHardTree = nx.DiGraph(
    [(0, 1), (2, 0), (2, 3), (3, 4), (5, 3), (6, 5), (7, 6), (8, 0), (8, 9), (9, 10), (11, 10), (12, 11), (13, 12),
     (8, 14), (15, 14), (16, 15), (15, 17), (17, 18), (18, 19)])
NoMajorityTree = nx.DiGraph(
    [(1, 2), (2, 3), (3, 4), (4, 5), (1, 0), (0, 6), (7, 6), (10, 7), (7, 8), (8, 9), (11, 0), (12, 11), (11, 13),
     (13, 14), (14, 15)])

T3Gap = getPath('111')  # Dual(T3Gap)=...
T3Gap.add_edges_from([('1l', '2l'), ('1l', '3l'), ('2l', '3l'), (0, '3l'), (0, '2l'), ('1l', 2)])
T3Gap.add_edges_from([('0r', '1r'), ('0r', '2r'), ('1r', '2r'), ('0r', 3), ('1r', 3), (1, '2r')])

N32 = getCycle('11100')
T234 = getCycle('11100')
T234.add_edges_from([(1, 5), (5, 2)])

N321 = getCycle('11100')
N321.add_edge(0, 3)

N123T3 = N321.copy()
N123T3.add_edges_from([(1, 5), (1, 6), (5, 6)])

N4321 = getCycle('1111000')
N4321.add_edge(0, 4)
N4321.add_edges_from([(0, 7), (7, 4)])

OrdMeetC2 = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (1, 3)])

if __name__ == "__main__":

    Graph = Ord  # T3#getCycle('101101')

    print(Graph.nodes)
    n = 4
    skip = 1
    maxSize = 20
    minSize = 3
    noLoop = True

    # G = pppower(Ord,('a0bcde', 'Abcde2'))
    # G = pppower(M7,('abC', 'bcA'))
    start = time.time()
    # G = getCore(G)
    # drawGraph(G)

    Gs = []
    fs = [('abc', '1AB')]
    # for k in range(2,n+1):
    #    fs += getReasonableFormulas(Graph,k)
    # fs = getReasonableFormulas(Graph, n, skip)
    # random.shuffle(fs)
    # print(len(fs))

    # gs = filterFormulasForACworks(Graph, fs, [PPG])
    # print(len(gs))

    if False:
        drawGraph(Graph, "graph")
        for f in fs:
            start = time.time()
            G = pppower(Graph, f)
            size = (len(G.nodes), len(G.edges))
            print(str((len(G.nodes), len(G.edges))) + " PPPower " + str(f) + " time " + str(time.time() - start))
            if len(G.nodes) < 500:
                start = time.time()
                new = True
                for H in Gs:
                    if ArcCon.isHomEqACworks(G, H):
                        new = False
                        break

                if new:
                    G = getCore(G)  # also makes loop test obsolete
                    Gs += [G]
                    print(f)
                    if len(G.nodes) <= maxSize and len(G.nodes) >= minSize:
                        time.sleep(0.1)
                        drawGraph(G, (f, size, (len(Graph.nodes), len(Graph.edges))))
                else:
                    print('#' + str(f))

    minSize = 1
    maxSize = 20
    skip = 1
    Graph = nx.DiGraph([(0, 1), (1, 0), (2, 3), (3, 2), (0, 2), (1, 3)])
    ACWorks = True
    R = [(0, 1, 2), (1, 4, 3)]
    fs = getReasonableFormulas(Graph, 5, 100)
    print(len(fs))
    if True:
        drawGraph(Graph, "graph")
        for f in fs:
            start = time.time()
            # G = pppowerB3(Graph, (f[0], f[1], R))
            G = pppower(Graph, f)
            size = (len(G.nodes), len(G.edges))
            print(str((len(G.nodes), len(G.edges))) + " PPPower " + str(f) + " time " + str(time.time() - start))
            if len(G.nodes) < 900:
                # if ACWorks:
                #    G = ArcCon.reduceHomComponentsACWorks(G)
                start = time.time()
                new = True
                for H in Gs:
                    if ArcCon.isHomEqACworks(G, H):
                        new = False
                        break

                if new:
                    G = getCore(G)  # also makes loop test obsolete
                    Gs += [G]
                    print(f)
                    if len(G.nodes) <= maxSize and len(G.nodes) >= minSize:
                        time.sleep(0.1)
                        drawGraph(G, (f, size, (len(Graph.nodes), len(Graph.edges))))
                else:
                    print('#' + str(f))

    #        if len(G.nodes) < 350:# just consider small graphs for performance reasons (getCore is slow otherwise)

    # G = pppower(T,("2a","A1"))
    # drawGraph(G)

    print(str("time " + str(time.time() - start)))


def solve(MNr):
    x = int(MNr[:3])
    y = int(MNr[:3:-1])
    x = int(MNr[2::-1])
    y = int(MNr[:3:-1])
    print('phi(' + str(x) + ') = ' + str(sympy.totient(x)))
    print('phi(' + str(y) + ') = ' + str(sympy.totient(y)))
    print('x^y mod 101 = ' + str(pow(x, y, 101)))
