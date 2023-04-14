import pandas as pd
from utils import get_generators, print_generators, h1_condition_getter_ternary, get_all_equations


#copiare il nome del db da utilizzare dentro la stringa
df = pd.read_csv('bulatov_db/algebre_con_monolito.csv', sep=';')
ternary = pd.read_csv('bulatov_db/funzioni_ternarie_quaternaria.csv', sep=';')
#inserire il nome delle relazioni di cui si vule
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



#term_list = get_all_equations(3, 3)






list_of_relations = ['0', '1', '2']

generators = get_generators(list_of_relations, df)

gen_unari, gen_binari, gen_ternari_quat = print_generators(generators)

h1_list = h1_condition_getter_ternary(ternary)

equation_dataframe = df = pd.DataFrame(h1_list, columns=['function', 'equation'])

equation_dataframe.to_csv('h1_equations.csv', index=False)

print()






