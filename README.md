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

It will build duckdb (`CMAKE_BINARY_DIR/lib/duckdb/install/lib/libduckdb`) and our library (`CMAKE_BINARY_DIR/libduckdb_library`) as shared library, so you can use them in your code. Make sure you copy the header files for duckdb (`CMAKE_BINARY_DIR/lib/duckdb/install/include`) and this library (`imputation/include`) and add them to your code.

## Usage

Use `ML_lib::register_functions (duckdb::ClientContext &context)` to add the functions introduced in this library in DuckDB

```
#include <duckdb.hpp>
#include <helper.h>

...

duckdb::DuckDB db(":memory:");
duckdb::Connection con(db);

ML_lib::register_functions(*con.context);
```

## Functions


* `to_cofactor(columns)`: Returns a triple of aggregates. Generates a triple data structure from a tuple. The data type of the columns indicate if it's a numerical (float) or categorical (integer) type.
* `sum_triple (triple)`: Aggregate function, returns a triple. Sums multiple triple aggregates and generate a resulting triple
* `sum_to_triple (columns)`: Aggregate function, returns a triple. Generates a triple aggregate which represents the sum of the table. Equivalent to `sum_triple(to_cofactor(columns))` but faster. The data type of the columns indicate if it's a numerical (float) or categorical (integer) type.
* `multiply_triple(triple)`: Multiply two triples together returning the resulting triple
* `lda_predict`
