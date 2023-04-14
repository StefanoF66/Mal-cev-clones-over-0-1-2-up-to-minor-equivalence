import pandas as pd
import numpy as np
from itertools import chain, combinations
from utils import get_generators, print_generators, h1_condition_getter_ternary, get_all_equations
df = pd.read_csv('bulatov_db/algebre_semplici.csv', sep=';')
ternary = pd.read_csv('bulatov_db/funzioni_ternarie_quaternaria.csv', sep=';')


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def miner_structures(df):
    df1 = df.drop(['Operazioni'], axis=1)
    list_vectors = df1.values.tolist()
    list_vectors.pop(0)
    s = list(powerset(list_vectors))
    s.pop(0)
    list_of_structures = []
    for gens in s:
        res = gens[0]
        for gen in gens:
            res = np.multiply(res, gen)
        list_of_structures.append(res)
    list_of_structures = [tuple(l) for l in list_of_structures]
    list_of_structures.append(tuple([1] * len(list_of_structures[0])))
    return set(list_of_structures)

structures_set = miner_structures(df)
structures = []
structure_list = [list(l) for l in list(structures_set)]
columns_mon = ['0', '1', '2', '01', '12', '02', 'psi2', 'tau1', 'tau2', 'psi’2', 'rho1', 'S01', 'T01', 'T1', 'T’1']
columns_sem = ['0', '1', '2', '01', '12', '02', 'T01', 'T02', 'T21', 'phi1', 'psi0', 'psi1', 'psi2', 'psi’0', 'psi’1',
               'psi’2', 'phi’1', 'phi’2', 'phi’3', 'phi’4', 'phi’5', 'phi’6']
df = pd.DataFrame(structure_list, columns=columns_sem)
df.to_csv('structures_simple.csv', index=False)
print('a')


