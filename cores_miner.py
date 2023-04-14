import pandas as pd
from pcsptools import Structure, polymorphisms, parse_identities, check_minor_condition
from pcsptools.solver import pyco_solver
import json
df_simple = pd.read_csv('structures_simple.csv', sep=',')
df_monolith = pd.read_csv('structures_with_monolith.csv', sep=',')



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

structures_list_sim = df_simple.values.tolist()
structures_list_mon = df_monolith.values.tolist()
structures = structures_list_mon
columns = columns_mon

poly_column = []

for structure_rel in structures:
    relations_list = [relations[rel] for (flag, rel) in zip(structure_rel, columns) if flag != 0]



    #selected_relations = ['0', '1', '2', '01', '02', '12', 'phi’1', 'phi’2', 'psi0', 'psi’0', 'psi’1', 'psi’2', 'T21']
    #selected_relations = ['S01']
    #relations_list = [relations[rel] for rel in selected_relations]


    structure = Structure(
            (0, 1, 2),
            relations_list,
            flag=True,
            )

    poly = polymorphisms(structure, structure, 1, solver=pyco_solver)
    poly_column.append([str(pol) for pol in poly])

df_monolith['un_poly'] = poly_column
df_monolith.to_csv('structures_mono_with_poly.csv', index=False)


