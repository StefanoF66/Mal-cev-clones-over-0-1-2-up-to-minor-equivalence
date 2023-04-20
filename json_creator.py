import json
relation = []


relations = ['0', '1', '2', '01', '12', '02', 'mi1', 'mi2', 'mi3',
             'phi0', 'phi1', 'phi2',
             'psi0', 'psi1', 'psi2',
             'tau1', 'tau2', 'tau3',
             'tau4', 'tau5', 'tau6',
             'psi’0', 'psi’1', 'psi’2',
             'phi’1', 'phi’2', 'phi’3',
             'phi’4', 'phi’5', 'phi’6',
             'rho1', 'rho2', 'rho3',
             'S01', 'S12', 'S20',
             'T01', 'T02', 'T21',
             'T1', 'T2', 'T3',
             'T’1',
             'T'
             ]

tuples = [((0)), ((1)), ((2)), ((0), (1)), ((1), (2)), ((2), (0)), ((0, 0), (1, 1), (2, 2), (0, 1), (1, 0)), ((0, 0), (1, 1), (2, 2), (0, 2), (2, 0)), ((0, 0), (1, 1), (2, 2), (1, 2), (2, 1)),
          ((0, 0), (1, 1), (2, 2)), ((0, 1), (1, 2), (2, 0)), ((0, 2), (1, 0), (2, 1)),
          ((0, 0), (1, 2), (2, 1)), ((0, 2), (1, 1), (2, 0)), ((0, 1), (1, 0), (2, 2)),
          ((0, 0), (1, 0), (2, 1)), ((0, 1), (1, 1), (2, 0)), ((0, 0), (1, 2), (2, 0)),
          ((0, 2), (1, 0), (2, 2)), ((0, 2), (1, 1), (2, 1)), ((0, 1), (1, 2), (2, 2)),
          ((1, 2), (2, 1)), ((0, 2), (2, 0)), ((0, 1), (1, 0)),
          ((0, 0), (1, 2)), ((0, 2), (1, 0)), ((0, 1), (1, 2)),
          ((0, 2), (1, 1)), ((0, 1), (2, 2)), ((0, 2), (2, 1)),
          ((0, 2), (1, 2), (2, 0), (2, 1)), ((0, 1), (2, 1), (1, 0), (1, 2)), ((0, 1), (0, 2), (1, 0), (2, 0)),
          ((0, 0, 0), (1, 0, 0), (2, 1, 0), (0, 1, 1), (1, 1, 1), (2, 0, 1)), ((1, 1, 1), (2, 1, 1), (0, 2, 1), (1, 2, 2), (2, 2, 2), (0, 1, 2)), ((2, 2, 2), (0, 2, 2), (1, 0, 2), (2, 0, 0), (0, 0, 0), (1, 2, 0)),
          ((0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1)),
          ((0, 0, 0, 0), (0, 0, 2, 2), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2), (2, 0, 2, 0), (2, 2, 0, 0), (2, 2, 2, 2)),
          ((2, 2, 2, 2), (2, 2, 1, 1), (2, 1, 2, 1), (2, 1, 1, 2), (1, 2, 2, 1), (1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 1, 1)),
          ((0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2)),
          ((0, 0, 0, 0), (0, 0, 2, 2), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2), (2, 0, 2, 0), (2, 2, 0, 0), (2, 2, 2, 2), (1, 1, 1, 1)),
          ((2, 2, 2, 2), (2, 2, 1, 1), (2, 1, 2, 1), (2, 1, 1, 2), (1, 2, 2, 1), (1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 1, 1), (0, 0, 0, 0)),
          ((0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1), (0, 0, 2, 2), (0, 1, 2, 2), (1, 0, 2, 2), (1, 1, 2, 2), (0, 2, 0, 2), (0, 2, 1, 2), (1, 2, 0, 2), (1, 2, 1, 2), (0, 2, 2, 0), (0, 2, 2, 1), (1, 2, 2, 0), (1, 2, 2, 1), (2, 0, 0, 2), (2, 0, 1, 2), (2, 1, 0, 2), (2, 1, 1, 2), (2, 0, 2, 0), (2, 0, 2, 1), (2, 1, 2, 0), (2, 1, 2, 1), (2, 2, 0, 0), (2, 2, 0, 1), (2, 2, 1, 0), (2, 2, 1, 1), (2, 2, 2, 2)),
          ((0, 0, 0, 0), (0, 0, 2, 1), (0, 0, 1, 2), (0, 1, 0, 1), (0, 1, 1, 0),  (0, 1, 2, 2), (1, 0, 0, 1), (1, 0, 1, 0),  (1, 0, 2, 2), (1, 1, 1, 1), (1, 1, 2, 0), (1, 1, 0, 2), (2, 0, 1, 1), (2, 0, 2, 0), (2, 0, 0, 2), (0, 2, 1, 1), (0, 2, 2, 0), (0, 2, 0, 2),
          (2, 1, 2, 1), (2, 1, 1, 2), (1, 2, 2, 1), (1, 2, 1, 2), (2, 2, 0, 1), (2, 2, 1, 0),  (2, 2, 2, 2))
          ]

dictionary = {relations[i]: tuples[i] for i in range(len(relations))}
with open('databases/relations.json', 'w') as fjson:
    json.dump(dictionary, fjson)
