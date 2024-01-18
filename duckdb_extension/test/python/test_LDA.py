import duckdb
import os
import pytest
import sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import KBinsDiscretizer


# Get a fresh connection to DuckDB with the duckdb_imputation extension binary loaded
@pytest.fixture
def in_data():
    extension_binary = "../../build/release/extension/duckdb_imputation/duckdb_imputation.duckdb_extension"#"os.getenv('DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
    conn.execute(f"INSTALL '{extension_binary}'")
    conn.execute(f"load duckdb_imputation")
    
    

    data = load_iris(as_frame=True, return_X_y=True)

    df_train, df_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.33, random_state=42)
    df_train["target"] = y_train
    df_test["target"] = y_test

    df_train = df_train.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})
    df_test = df_test.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})

    df_train = df_train.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    df_test = df_test.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    
    conn.sql('create table iris_train (id integer primary key, s_length float, s_width float, p_length float, p_width float, target integer);')
    conn.sql("INSERT INTO iris_train SELECT * FROM df_train")

    conn.sql('create table iris_test (id integer primary key, s_length float, s_width float, p_length float, p_width float, target integer);')
    conn.sql("INSERT INTO iris_test SELECT * FROM df_test")

    
    return [conn,df_train,df_test]

@pytest.fixture
def in_data_cat():
    extension_binary = "../../build/release/extension/duckdb_imputation/duckdb_imputation.duckdb_extension"#"os.getenv('DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
    conn.execute(f"INSTALL '{extension_binary}'")
    conn.execute(f"load duckdb_imputation")
    
    
    
    data = load_iris(as_frame=True, return_X_y=True)

    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform', subsample=None)
    df = data[0].rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})
    train_data = est.fit_transform(df[["s_length", "s_width", "p_length"]])
    df["s_length"] = train_data[:, 0]
    df["s_width"] = train_data[:, 1]
    df["p_length"] = train_data[:, 2]

    df_train, df_test, y_train, y_test = train_test_split(df, data[1], test_size=0.33, random_state=42)
    df_train["target"] = y_train
    df_test["target"] = y_test

    df_train = df_train.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})
    df_test = df_test.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})

    df_train = df_train.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    df_test = df_test.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    
    conn.sql('create table iris_train (id integer primary key, s_length integer, s_width integer, p_length integer, p_width float, target integer);')
    conn.sql("INSERT INTO iris_train SELECT * FROM df_train")

    conn.sql('create table iris_test (id integer primary key, s_length integer, s_width integer, p_length integer, p_width float, target integer);')
    conn.sql("INSERT INTO iris_test SELECT * FROM df_test")
    
    
    df_encoded = df.rename(columns={"sepal length (cm)": "s_length", "sepal width (cm)": "s_width", "petal length (cm)": "p_length", "petal width (cm)": "p_width"})
    df_encoded = df_encoded.reset_index(drop=True).reset_index().rename(columns={'index':'id'})
    df_encoded = pd.get_dummies(df_encoded, columns=['s_length','s_width','p_length'])
    df_encoded["target"] = data[1]
    
    df_train_encoded, df_test_encoded = train_test_split(df_encoded, test_size=0.33, random_state=42)
    
    return [conn,df_train,df_test,df_train_encoded, df_test_encoded]


def test_lda_no_norm_cat(in_data_cat):
    conn = in_data_cat[0]
    df_train = in_data_cat[1]
    df_test = in_data_cat[2]
    df_train_encoded=in_data_cat[3]
    df_test_encoded=in_data_cat[4]
    
    
    triple = conn.sql("SELECT sum_to_triple_1_4(p_width, s_length, s_width, p_length, target) as agg from iris_train;").fetchall()
    str_triple = str(triple[0][0])
    cast = "::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][])"
    query = "select lda_train("+str_triple+cast+", 3, 0.01, false)"
    print(query)
    params = conn.sql(query).fetchall()

    
    predict = conn.sql("SELECT id, lda_predict ("+str(params[0][0])+"::float[], false, p_width, s_length, s_width, p_length) as pred from iris_test").df()
    
    df_test = df_test[["id", "target"]].merge(predict, left_on='id', right_on='id')
    acc_ddb = accuracy_score(df_test["target"], df_test["pred"])
    
    clf = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage=0)
    clf.fit(df_train_encoded.drop(["target", "id"], axis=1), df_train_encoded["target"])
    print("Accuracy SKLearn LDA: ", clf.score(df_test_encoded.drop(["target", "id"], axis=1), df_test["target"]))
    acc_sklearn = clf.score(df_test_encoded.drop(["target", "id"], axis=1), df_test["target"])
    
    assert (round(acc_ddb, 3) == round(acc_sklearn, 3))

    

def test_lda_norm_cat(in_data_cat):
    conn = in_data_cat[0]
    df_train = in_data_cat[1]
    df_test = in_data_cat[2]
    df_train_encoded=in_data_cat[3]
    df_test_encoded=in_data_cat[4]
    
    
    triple = conn.sql("SELECT sum_to_triple_1_4(p_width, s_length, s_width, p_length, target) as agg from iris_train;").fetchall()
    str_triple = str(triple[0][0])
    cast = "::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][])"
    query = "select lda_train("+str_triple+cast+", 3, 0.01, true)"
    print(query)
    params = conn.sql(query).fetchall()

    
    predict = conn.sql("SELECT id, lda_predict ("+str(params[0][0])+"::float[], true, p_width, s_length, s_width, p_length) as pred from iris_test").df()
    
    df_test = df_test[["id", "target"]].merge(predict, left_on='id', right_on='id')
    acc_ddb = accuracy_score(df_test["target"], df_test["pred"])
    
    clf = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage=0)
    clf.fit(df_train_encoded.drop(["target", "id"], axis=1), df_train_encoded["target"])
    print("Accuracy SKLearn LDA: ", clf.score(df_test_encoded.drop(["target", "id"], axis=1), df_test["target"]))
    acc_sklearn = clf.score(df_test_encoded.drop(["target", "id"], axis=1), df_test["target"])
    
    assert (round(acc_ddb, 3) == round(acc_sklearn, 3))
    
    
def test_lda_no_norm(in_data):
    conn = in_data[0]
    df_train = in_data[1]
    df_test = in_data[2]
    
    triple = conn.sql("SELECT sum_to_triple_4_1(s_length, s_width, p_length, p_width, target) as agg from iris_train;").fetchall()
    str_triple = str(triple[0][0])
    cast = "::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][])"
    query = "select lda_train("+str_triple+cast+", 0, 0, false)"
    params = conn.sql(query).fetchall()
    predict = conn.sql("SELECT id, lda_predict ("+str(params[0][0])+"::float[], false, s_length, s_width, p_length, p_width) as pred from iris_test").df()
    
    df = df_test[["id", "target"]].merge(predict, left_on='id', right_on='id')
    acc_ddb = accuracy_score(df["target"], df["pred"]);
    print("Accuracy DuckDB LDA: ", acc_ddb)
    clf = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage=0)
    clf.fit(df_train.drop(["target", "id"], axis=1), df_train["target"])
    acc_sklearn = clf.score(df_test.drop(["target", "id"], axis=1), df_test["target"])
    
    print("Accuracy LDA SKLearn ", acc_sklearn)
    
    assert (round(acc_ddb, 3) == round(acc_sklearn, 3))

def test_lda_norm(in_data):
    conn = in_data[0]
    df_train = in_data[1]
    df_test = in_data[2]
    
    triple = conn.sql("SELECT sum_to_triple_4_1(s_length, s_width, p_length, p_width, target) as agg from iris_train;").fetchall()
    str_triple = str(triple[0][0])
    cast = "::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][])"
    query = "select lda_train("+str_triple+cast+", 0, 0, true)"
    params = conn.sql(query).fetchall()
    predict = conn.sql("SELECT id, lda_predict ("+str(params[0][0])+"::float[], true, s_length, s_width, p_length, p_width) as pred from iris_test").df()
    
    df = df_test[["id", "target"]].merge(predict, left_on='id', right_on='id')
    acc_ddb = accuracy_score(df["target"], df["pred"]);
    print("Accuracy DuckDB LDA: ", acc_ddb)
    clf = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage=0)
    clf.fit(df_train.drop(["target", "id"], axis=1), df_train["target"])
    acc_sklearn = clf.score(df_test.drop(["target", "id"], axis=1), df_test["target"])
    
    print("Accuracy LDA SKLearn ", acc_sklearn)
    
    assert (round(acc_ddb, 3) == round(acc_sklearn, 3))
