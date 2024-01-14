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

This repository includes an extension for DuckDB 0.9 and a C++ library. The DuckDB extension implements ring aggregates and ML models. The C++ library implements imputation algorithms and some utilities, and it's optional.


## Build the extension 


1. Clone this repository.
2. Clone DuckDB 0.9.2 from its repository and place it in duckdb_extension/duckdb. If you want to use another version, clone the specific version, but the library might not work.

	```
cd duckdb_extension
git clone https://github.com/duckdb/duckdb.git
cd duckdb
git checkout 5ec85a719940a9fade15c38e7601712e9cef58d8 (9.2 3c695d7ba94d95d9facee48d395f46ed0bd72b46)
git clean -df
	```

3. **Only if you want to use the imputation functions:** apply the patch inside the repository.
This patch will replace the SQL command


	```
	ALTER TABLE ... ALTER COLUMN ... SET DEFAULT ...
	```
	in DuckDB with a custom operator. **Make sure you won't need it. You'll also need to always use this specific build of DuckDB for imputation functions.**

	```
	cd duckdb
	git apply ../../duckdb_imputation.patch
	```
	
4. Build the extension

	```
	make
	```
		
	The main binaries that will be built are:
	
	```sh
	./build/release/duckdb
	./build/release/test/unittest
	./build/release/extension/duckdb_imputation/	duckdb_imputation.duckdb_extension
	```

	- `duckdb` is the binary for the duckdb shell with the extension code automatically loaded.
	- `unittest` is the test runner of duckdb. Again, the extension is already linked into the binary.
	- `duckdb_imputation.duckdb_extension` is the loadable binary.

### Install the extension

**Make sure the DuckDB version you used to compile the code and the version you want to use match!**

To install the extension binaries, you will need to do two things. Firstly, DuckDB should be launched with the
`allow_unsigned_extensions` option set to true. How to set this will depend on the client you're using. Some examples:

CLI:

```shell
duckdb -unsigned
```

Python:

```python
con = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
```

NodeJS:

```js
db = new duckdb.Database(':memory:', {"allow_unsigned_extensions": "true"});
```

Then, to install and load the extension binaries, in the DuckDB SQL run:

```
INSTALL 'path/to/duckdb_imputation.duckdb_extension';
LOAD duckdb_imputation;
```

The extension path is usually 

```
<repo_path>/duckdb_extension/build/debug/extension/duckdb_imputation/duckdb_imputation.duckdb_extension
```

Optionally, run the tests with `make test_python`. You need python with DuckDB 0.9.2 (pip install duckdb==0.9.2 and pytest.


## Usage

The extension already includes all the function required to generate the aggregate values and use the implemented models.

### Functions

All the function which accept a varying number of columns accept maximum 1024 columns, and numerical columns must be before categorical columns. Numerical columns must be of float type, while the type of categorical columns must be integer.

* `to_cofactor(columns)`: Returns a triple of aggregates for each tuple. The data type of the columns indicate if it's a numerical (float) or categorical (integer) type.
* `to_nb_agg(columns)`: Returns the aggregate for Naive Bayes (continuous/categorical) for each tuple.


* `sum_triple (triple)`: Aggregate function, returns a triple. Sums multiple triple aggregates and generates a resulting triple
* `sum_nb_agg (triple)`: Aggregate function, returns aggregates for Naive Bayes. Sums multiple Naive Bayes aggregates and generates the final value


* `sum_to_triple_x_y (columns)`: Aggregate function, returns a triple. Replace x and y with the number of numerical and categorical columns you want to pass (up to 20 numerical and 20 categorical). Generates a triple aggregate which represents the sum aggregate of the table. Equivalent to `sum_triple(to_cofactor(columns))` but faster.
* `sum_to_nb_agg_x_y`: Aggregate function, returns a Naive Bayes aggregate. Replace x and y with the number of numerical and categorical columns you want to pass (up to 20 numerical and 20 categorical). Generates a triple aggregate which represents the sum aggregate of the table. Equivalent to `sum_nb_agg(to_nb_agg(columns))` but faster.


* `multiply_triple(triple1, triple2)`: Multiply two triples together returning the resulting triple
* `multiply_nb_agg(nb_agg_1, nb_agg_2)`: Multiply two Naive Bayes aggregates together returning the resulting triple

### Example

The following query computes a triple of aggregates over a join. A triple of aggregates for each table is computed at the beginning, grouping by the join key. They are then multiplied together and summed to compute the final aggregate.

```
select sum_triple(multiply_triple(A,B)) FROM 
	(SELECT gb as gb, sum_to_triple_2_2(b,c,d,e) AS A 
		FROM test1 GROUP BY gb) as a 
	INNER JOIN 
	(SELECT gb as gb, sum_to_triple_2_2(a,c,d,f) AS B 
		FROM test2 GROUP BY gb) as b 
	on a.gb = b.gb;
	
```

In the test directory you can find additional examples.

