The 3 directories in this project provide the databases concerning the Mal'cev clones on a 3-element set up to minor equivalence.

In the directory "Standard_rlations" we have a json file containing all the relations 
used by A.Bulatov in \cite{} to characterize the Mal'cev clones on a 3-element.

In the directory "Functional_generators" we have 2 files "algebras_with_a_monolith.csv" and "simple_algebras.csv". In these files 
we have all the functional generators of the polymorphisms clone of structures of with a given congruence lattice. The tables list those generators showing 
which standard relations they preserve. For a given structure R = (R_1,...,R_n), the functional of Pol(R) can be recovered querying the dataset and selecting 
those lines with a 1 for every column R_i, with i = 1,...,n

In "Functional_generators" with further have the multiplication tables of all the functional generators in the csv files "unary_functions.csv", 
"binary_functions.csv", "3-ary_and_4-ary_functions.csv" split accordingly to their arity. In the last one the only 4-ary functional generator is listed as a 
ternary function of the first component, i.e., the multiplication table of f232_i is the multiplication table of f232 for the tuple (i,x_1,x_2,x_3) 
with x_1,x_2,x_3 = 0,1,2

The last directory "Structures_databases" relational structure invariant of all Mal'cev clones on a 3-element set divide by their congruence lattices. 
In the case with a monolith, the monolith {01}{2} is always assumed to be in the invariant and not written. In each of the two congruence case we have two files.
The first is simply listing the structures, providing some information about standard relations included, unary polymorphisms, cardinality of the core, minor equivalence 
class of the cores in the case they are not a core, h-1 conditions satisfied (relatively to the characterization of the paper), minor equivalence class.

The second file is filtering the first one, listing the structures that are idempotent and cores (the ones that are most relevant to our paper)
