# **Mal'cev Clones on a 3-Element Set: Database and Resources**  

This repository contains databases related to **Mal'cev clones** on a 3-element set, classified up to **minor equivalence**. The data is organized into three directories, each providing different aspects of the classification.  

## **1. Standard Relations**  
The `Standard_relations` directory contains a JSON file that lists all relations used by **A. Bulatov** in [REFERENCE] to characterize Mal'cev clones on a 3-element set.  

## **2. Functional Generators**  
The `Functional_generators` directory includes data on functional generators of polymorphism clones, structured as follows:  

- **`algebras_with_a_monolith.csv`** and **`simple_algebras.csv`**:  
  These files list all functional generators of polymorphism clones corresponding to structures with a given congruence lattice. Each table provides:  
  - The functional generators.  
  - The standard relations they preserve.  

  To extract the functional generators of **Pol(R)** for a given structure \( R = (R_1, \dots, R_n) \), query the dataset and select all rows where the value in columns corresponding to \( R_i \) (for \( i = 1, \dots, n \)) is **1**.  

- **Multiplication Tables**:  
  This directory also contains CSV files with the multiplication tables of all functional generators, categorized by arity:  
  - **`unary_functions.csv`**  
  - **`binary_functions.csv`**  
  - **`3-ary_and_4-ary_functions.csv`**  

  The last file includes the only **4-ary functional generator**, which is represented as a **ternary function** in its first component. Specifically, the multiplication table of \( f232_i \) corresponds to the values of \( f232 \) evaluated on the tuple \( (i, x_1, x_2, x_3) \) for \( x_1, x_2, x_3 \in \{0,1,2\} \).  

## **3. Structure Databases**  
The `Structures_databases` directory contains **relational structure invariants** for all Mal'cev clones on a 3-element set, categorized by **congruence lattices**.  

- In cases where a **monolith** exists, it is always assumed to be \( \{\{0,1\}, \{2\}\} \) and is not explicitly listed.  
- For each congruence type, two files are provided:  
  1. A list of structures with:  
     - Standard relations included.  
     - Unary polymorphisms.  
     - Core cardinalities.  
     - Minor equivalence classes of the cores (when the structure is not already a core).  
     - \( h-1 \) conditions satisfied (as per the characterization in [REFERENCE]).  
     - Minor equivalence classification.

   2. The first list restricted to idempotent 3-element cores (the ones that are most relevant for our paper).

---

This structured dataset provides a valuable resource for researchers studying Mal'cev clones and their functional representations.

