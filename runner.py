from pcsptools import Structure, polymorphisms, parse_identities, check_minor_condition
from pcsptools.solver import pyco_solver
import json

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

#selected_relations = ['0', '1', '2', '01', '02', '12', 'phi’1', 'phi’2', 'psi0', 'psi’0', 'psi’1', 'psi’2', 'T21']
selected_relations = ['S01']
relations_list = [relations[rel] for rel in selected_relations]


structure = Structure(
        (0, 1, 2),
        relations_list,
        flag=True,
        )

# structure = Structure(
#         (0, 1, 2),
#         relations_list[0]
#         #relations_list[1]
#         )

# structure = Structure(
#         (0, 1, 2),
#         ((0,1),(2,0)),
#         ((1,1),(2,2))
#         )

#poly = polymorphisms(structure2, structure2, 1, solver=pyco_solver)

solution = check_minor_condition(
        structure, structure,
        parse_identities(
            "s(xxy) = s(xyx)"))

if solution is not None:
    print(solution)
else:
    print('No such polymorphisms!')

# for pol in poly:
#     print(pol)
#
# print(poly)