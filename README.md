This repository contains the implementation of ... based on the paper:...

This repository contains both a library for performing efficient Machine Learning and code to run MICE inside DuckDB. The supported models are:

```
Linear Regression
Stochastic Linear Regression
Linear Discriminant Analysis
Quadratic Discriminant Analysis
Naive Bayes (Gaussian and categorical)
```

At the moment, imputation is always done with Stochastic Linear Regression (continuous columns) and Linear Discriminant Analysis (categorical columns).

## Installation 

Clone this repository.

Clone DuckDB 0.9 from its repository and place it in lib/duckdb.

```
mkdir lib
cd lib
git clone https://github.com/duckdb/duckdb.git
cd duckdb
git checkout 5ec85a719940a9fade15c38e7601712e9cef58d8
git clean -df
```

**This library will not work with the version of DuckDB available on the website, and might not work with different versions of DuckDB** as it relies on some internal functions.


**Only if you want to use the imputation functions:** apply the patch inside the repository to duckdb with
`git apply duckdb_imputation.patch`

This patch will replace the SQL command
```ALTER TABLE ... ALTER COLUMN ... SET DEFAULT ...```
with a custom operator. **Make sure you won't need it.**

Compile with

```
cmake .
make
```
