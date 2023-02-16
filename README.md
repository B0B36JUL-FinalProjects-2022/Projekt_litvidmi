# SysRec
My implementation of simple algorithms for collaborative filtering. 

1) Matrix factorization - try to recreate the original user-item interaction (rating) matrix by multiplying user-category and category-item matrices. 

2) Autorec - try to encode a given sparse column/row of original user-item interaction matrix into a vector with non-zero values.

Necessary data-preprocessing tools are also provided.
Aimed primarily at Movielens dataset.

# Installation
``` julia
add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_litvidmi
```

# Usage
/examples directory contains detailed usage scenarios
for each model.

# TODO
ALSW as an optimization algorithm.
Implementation of factorization machines and 
evaluation metrics for bayesian personalized ranking.