import pandas as pd
import itertools
import torch
import string
from pcsptools.polymorphisms import LabelCover


'''
Get generators from a list of generators
'''
def get_generators(list_of_relations, df):
    ret_df = df
    for rel in list_of_relations:
        ret_df = ret_df[ret_df[rel] == 1]

    return ret_df

'''
print generators
'''
def print_generators(generators):
    list_gen = generators['Operazioni'].tolist()
    df_unarie = pd.read_csv('bulatov_db/funzioni_unarie.csv', sep=';')
    df_binarie = pd.read_csv('bulatov_db/funzioni_binarie.csv', sep=';')
    df_ternarie_quat = pd.read_csv('bulatov_db/funzioni_ternarie_quaternaria.csv', sep=';')
    df_unarie = df_unarie[df_unarie['Operazioni'].isin(list_gen)]
    df_binarie = df_binarie[df_binarie['Operazioni'].isin(list_gen)]
    df_ternarie_quat = df_ternarie_quat[df_ternarie_quat['Operazioni'].isin(list_gen)]

    return df_unarie, df_binarie, df_ternarie_quat

'''
print h_1 mal'cev conditions satisfied by a list of terms
'''
def h1_condition_getter_ternary(terms):
    cycles = ['(x,y,z)', '(x,z,y)', '(y,x,z)', '(y,z,x)', '(z,y,x)', '(z,x,y)']
    cycles_num = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1)]
    collapses = ['(x,y,y)', '(y,x,x)', '(x,x,y)', '(y,y,x)', '(x,y,x)', '(y,x,y)']
    collapses_num = [(0, 1, 1), (1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 0), (1, 0, 1)]
    var_list = ['xxx', 'xxy', 'xxz', 'xyx', 'xyy', 'xyz', 'xzx', 'xzy', 'xzz', 'yxx', 'yxy', 'yxz', 'yyx', 'yyy', 'yyz', 'yzx', 'yzy', 'yzz', 'zxx', 'zxy', 'zxz', 'zyx', 'zyy', 'zyz', 'zzx', 'zzy', 'zzz']
    h1eq_list = []

    for (index, gen) in terms.iterrows():
        gen = gen.tolist()
        function = gen.pop(0)
        gen = torch.tensor(gen)
        gen = torch.reshape(gen, (3, 3, 3))
        for (i, elem) in enumerate(cycles_num[:-1]):
            for (j, elem2) in enumerate(cycles_num[i+1:]):
                if all(gen[a[elem[0]], a[elem[1]], a[elem[2]]] == gen[a[elem2[0]], a[elem2[1]], a[elem2[2]]] for a in itertools.product(range(3), range(3), range(3))):
                    h1eq_list.append((function, function+cycles[i]+'='+function+cycles[i+1+j]))

        for (i, elem) in enumerate(collapses_num[:-1]):
            for (j, elem2) in enumerate(collapses_num[i+1:]):
                if all(gen[a[elem[0]], a[elem[1]], a[elem[2]]] == gen[a[elem2[0]], a[elem2[1]], a[elem2[2]]] for a in itertools.product(range(3), range(3), range(3))):
                    h1eq_list.append((function, function+collapses[i]+'='+function+collapses[i+1+j]))

    return h1eq_list


def get_tuples_of_variables(arity, terms):
    if arity > 30 or terms > 25:
        print('Error: too many terms or arity too much')
        return None
    term_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
                 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w'
                 , 23: 'x', 24: 'y', 25: 'z'}

    chars = string.ascii_letters + string.digits# "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = chars[:arity]
    tuples_list = [''.join(var_tuple) for var_tuple in itertools.product(chars, repeat=arity)]

    term_list = [term_dict[i] +"("+variables+")" for (i, variables) in itertools.product(range(terms), tuples_list)]
    return term_list


def get_all_equations(arity, terms):
    tuples_list = get_tuples_of_variables(arity, terms)
    equations_list = [term1+' = '+term2 for (term1, term2) in itertools.product(tuples_list, tuples_list)]
    comb_equations = []
    for i in range(len(equations_list) + 1):
        # Generating sub list
        comb_equations += [list(j) for j in itertools.combinations(equations_list, i)]
    return comb_equations


def parse_identities(*args):
    """ parses identities from strings to an LC instance:
        each arg contains a bunch of linked identities, e.g.
        'm(x, x, y) = m(x, y, x) = m(y, x, x)', or
        '   s(123,123)=s(231,321) '"""
    fs, constraints = dict(), list()
    if isinstance(args[0], list):
        args = tuple(args[0])

    for id_no, line in enumerate(args):
        terms, xs = [], set()
        state, fargs = 1, ""
        for i, char in enumerate(line):
            if state == 0:
                if char == ' ':
                    continue
                if char == '=':
                    state = 1
                else:
                    raise ValueError(
                        f"Unexpected character '{char}' at {id_no}:{i}.")
            elif state == 1:
                if char in " ":
                    continue
                if char in string.ascii_letters:
                    f = char
                    state = 2
                else:
                    raise ValueError(
                        f"Unexpected character '{char}' at {id_no}:{i}.")
            elif state == 2:
                if char == ' ':
                    continue
                if char == '(':
                    state = 3
            elif state == 3:
                if char in " ,":
                    continue
                if char in string.ascii_letters or char in string.digits:
                    fargs += char
                    xs.add(char)
                elif char == ")":
                    if len(fargs) == 0:
                        raise ValueError(
                            f"Unexpected character '{char}' at {id_no}:{i}.")
                    if f in fs and fs[f] != len(fargs):
                        raise ValueError(
                            f"'{f}' has ambiguous arity.")
                    else:
                        fs[f] = len(fargs)
                    terms.append((f, fargs[:]))
                    fargs = ""
                    state = 0
                else:
                    raise ValueError(
                        f"Unexpected character '{char}' at {id_no}:{i}.")
        if state == 1:
            raise ValueError("Trailing '='?")
        elif state == 2:
            raise ValueError("Expected '(', but the string ended.")
        elif state == 3:
            raise ValueError("Expected ')', but the string ended.")
        elif len(terms) == 1:
            raise ValueError("One term does not form an equation!")
        elif state != 0:
            raise ValueError("Unexpected state at the end of string.")

        fs[f'i{id_no}'] = len(xs)
        x_to_i = {x: i for i, x in enumerate(xs)}

        for f, fargs in terms:
            constraints.append((
                (f, f'i{id_no}'),
                tuple((i, x_to_i[x]) for i, x in enumerate(fargs))))

    return LabelCover(fs.items(), constraints)








