import pandas as pd
import re
import numpy as np
from itertools import chain, combinations
df_mon = pd.read_csv('structures_mono_with_poly.csv', sep=',')
#df_mon = pd.read_csv('structures_simple_with_poly.csv', sep=',')
l1 = df_mon['un_poly'].values.tolist()
l1 = [list(l.replace('[', '').replace(']', '').split('\',')) for l in l1]
l1 = [[re.sub(r'[^0-9]', '', l) for l in l2] for l2 in l1]
df_mon['un_poly'] = l1
#l3 = df_sim['un_poly'].values.tolist()
#l3 = [list(l.replace('[', '').replace(']', '').split('\',')) for l in l3]
#l3 = [[re.sub(r'[^0-9]', '', l) for l in l2] for l2 in l3]
#df_sim['un_poly'] = l3
#query_col = [1 if '001020' in l else 0 for l in df_sim['un_poly'].values.tolist()]
#df_sim['query'] = query_col
#df_sim[df_sim['query'] == 1]
query_col = ['Top' if '001020' in l or '011121' in l or '021222' in l else 0 for l in df_mon['un_poly'].values.tolist()]
df_mon['Category'] = query_col
struct_coll = [cat if cat != 0 else 'binary' if '001120' in l or '001121' in l or '001222' in l or '001022' in l or
                                                '011122' in l or '021122' in l else 'core' for (l, cat) in
               zip(df_mon['un_poly'].values.tolist(), df_mon['Category'].values.tolist())]
df_mon['Category'] = struct_coll
#df_sim[df_sim['Category'] == 'Ternary']
df_mon.to_csv('structures_mon_with_poly_cores.csv', index=False)


print('a')



