import pandas as pd
from pcsptools import Structure, check_minor_condition
import ast
import json
from utils import parse_identities
df_monolith = pd.read_csv('databases/h1_mon_db.csv', sep=',')
df_simple = pd.read_csv('databases/h1_sim_db.csv', sep=',')


with open('databases/relations.json') as json_file:
    relations = json.load(json_file)

'''
Per algebre semplici: 
Sottalgebre:
'0' '1' '2' '01' '12' '02'
Automorfismi:
'fi1' 'psi0' 'psi1' 'psi2'
Internal-iso:
'psi’0' 'psi’1' 'psi’2' 'phi’1' 'phi’2' 'phi’3' 'phi’4' 'phi’5' 'phi’6'
Type of 2 - element subalgebra:
'T01' 'T02' 'T12'
**Per algebre con monolito**:
Sottalgebre:
'0' '1' '2' '01' '12' '02'
Automorfismi:
'fi2'
Homo in 2 - element sub:
'tau1' 'tau2'
Internal iso:
'fi’2'
Automorfismi di A / mi:
'rho1'
Linear dep:
'S01'
Type of 2 - element subalgebra:
'T01' 'T1' 'T’1'
'''

with open('databases/relations.json') as json_file:
    relations = json.load(json_file)

columns_mon = ['0', '1', '2', '01', '12', '02', 'psi2', 'tau1', 'tau2', 'psi’2', 'rho1', 'S01', 'T01', 'T1', 'T’1']
columns_sim = ['0', '1', '2', '01', '12', '02', 'T01', 'T02', 'T21', 'phi1', 'psi0', 'psi1', 'psi2', 'psi’0', 'psi’1',
               'psi’2', 'phi’1', 'phi’2', 'phi’3', 'phi’4', 'phi’5', 'phi’6']

h1_df = pd.read_csv('databases/h1_identities.csv')
h1_list = h1_df.values.tolist()
h1_con_name = [l[0] for l in h1_list]
h1_identities = [ast.literal_eval(l[1]) for l in h1_list]
h1_identities = [[[fun[:1] + '(' + fun[1:] + ')'for fun in equ] for equ in eqs] for eqs in h1_identities]
h1_identities = [[' = '.join(l) for l in eqs] for eqs in h1_identities]

structures_list_sim = df_simple.drop(['un_poly'], axis=1).values.tolist()
structures_list_mon = df_monolith.values.tolist()
paths = ['databases/h1_sim_db.csv', 'databases/h1_mon_db.csv']
structures_list = [structures_list_sim, structures_list_mon]
columns_list = [columns_sim, columns_mon]

#SELEZIONARE LE IMPOSTAZIONI

#h1_con_name = ['malcev', 'gumm'] #['malcev', 'gumm']
#h1_identities = ["m(xxy) = m(yxx) = m(yyy)", "s(xy) = s(yx)"] #["m(xxy) = m(yxx) = m(yyy)", "s(xy) = s(yx)"]

for (path, structures, columns) in zip(paths, structures_list, columns_list):
    selected_dataframe = pd.read_csv(path, sep=',')
    for (m_cond, identities) in zip(h1_con_name[4], h1_identities[4]):
        structure_type = []
        for structure_rel in structures:
            structure_rel = structure_rel[0:len(columns)]
            relations_list = [relations[rel] for (flag, rel) in zip(structure_rel, columns) if flag != 0]
            structure = Structure(
                    (0, 1, 2),
                    relations_list,
                    flag=True,
                    )
            # solution = check_minor_condition(
            #     structure, structure,
            #     parse_identities('m(xxy) = m(yxx)', 's(xy) = s(yx)'))
            solution = check_minor_condition(
                        structure, structure,
                        parse_identities(identities))
            if solution is not None:
                structure_type.append(1)
            else:
                structure_type.append(0)
            # flag = 1
            # for identity in identities:
            #     solution = check_minor_condition(
            #         structure, structure,
            #         parse_identities(identity))
            #     if solution is not None:
            #         #structure_type.append(1)
            #         continue
            #     else:
            #         structure_type.append(0)
            #         flag = 0
            #         break
            # if flag == 1:
            #     structure_type.append(1)
        selected_dataframe[m_cond] = structure_type
    selected_dataframe.to_csv(path, index=False)