import pandas as pd
from pcsptools import Structure, polymorphisms, parse_identities, check_minor_condition
from pcsptools.solver import pyco_solver
import json
df_monolith = pd.read_csv('structures_mon_with_poly_cores.csv', sep=',')
df_simple = pd.read_csv('structures_sim_with_poly_cores.csv', sep=',')

with open('relations.json') as json_file:
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

columns_mon = ['0', '1', '2', '01', '12', '02', 'psi2', 'tau1', 'tau2', 'psi’2', 'rho1', 'S01', 'T01', 'T1', 'T’1']
columns_sim = ['0', '1', '2', '01', '12', '02', 'T01', 'T02', 'T21', 'phi1', 'psi0', 'psi1', 'psi2', 'psi’0', 'psi’1',
               'psi’2', 'phi’1', 'phi’2', 'phi’3', 'phi’4', 'phi’5', 'phi’6']

structures_list_sim = df_simple.drop(['un_poly'], axis=1).values.tolist()
structures_list_mon = df_monolith.drop(['un_poly'], axis=1).values.tolist()
structures = structures_list_sim
columns = columns_sim

structure_type = []

for structure_rel in structures:
    if structure_rel[-1] == 'binary':
        structure_rel.pop()
        relations_list = [relations[rel] for (flag, rel) in zip(structure_rel, columns) if flag != 0]
        structure = Structure(
                (0, 1, 2),
                relations_list,
                flag=True,
                )

        solution = check_minor_condition(
            structure, structure,
            parse_identities(
                "s(xy) = s(yx)"))
        if solution is not None:
            structure_type.append('[q, m]')
            continue

        solution = check_minor_condition(
            structure, structure,
            parse_identities(
                "s(xxy) = s(yxx) = s(xyx) = s(xxx)"))
        if solution is not None:
            structure_type.append('[d_3, m]')
        else:
            structure_type.append('[m]')
    elif structure_rel[-1] == 'Top':
        structure_type.append('Top')
    else:
        structure_type.append('3-core')

df_simple['Class'] = structure_type
df_simple.to_csv('structures_sim_classes.csv', index=False)


print('a')