import duckdb
import os
import pytest

# Get a fresh connection to DuckDB with the duckdb_imputation extension binary loaded
@pytest.fixture
def duckdb_conn():
    extension_binary = "../../build/release/extension/duckdb_imputation/duckdb_imputation.duckdb_extension" #os.getenv('DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH')
    #if (extension_binary == ''):
    #    raise Exception('Please make sure the `DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
    conn.execute(f"INSTALL '{extension_binary}'")
    conn.execute(f"load duckdb_imputation")

    conn.execute("CREATE TABLE test(gb INTEGER, a FLOAT, b FLOAT, c FLOAT, d INTEGER, e INTEGER, f INTEGER);")
    conn.execute("INSERT INTO test VALUES (1,1,2,3,4,5,6), (1,5,6,7,8,9,10), (2,2,1,3,4,6,8), (1,5,7,6,8,10,12), (2,2,1,3,4,6,8)")

    return conn



def test_lift_all(duckdb_conn):
    duckdb_conn.execute("SELECT to_cofactor(a,b,c,d,e,f) from test")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 1, 'lin_num': [1.0, 2.0, 3.0], 'quad_num': [1.0, 2.0, 3.0, 4.0, 6.0, 9.0], 'lin_cat': [[{'key': 4, 'value': 1.0}], [{'key': 5, 'value': 1.0}], [{'key': 6, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 1.0}], [{'key': 5, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 4, 'value': 2.0}], [{'key': 5, 'value': 2.0}], [{'key': 6, 'value': 2.0}], [{'key': 4, 'value': 3.0}], [{'key': 5, 'value': 3.0}], [{'key': 6, 'value': 3.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}], [{'key1': 4, 'key2': 5, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}], [{'key1': 5, 'key2': 5, 'value': 1.0}], [{'key1': 5, 'key2': 6, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}]]}"))
    assert(res[1][0] == eval("{'N': 1, 'lin_num': [5.0, 6.0, 7.0], 'quad_num': [25.0, 30.0, 35.0, 36.0, 42.0, 49.0], 'lin_cat': [[{'key': 8, 'value': 1.0}], [{'key': 9, 'value': 1.0}], [{'key': 10, 'value': 1.0}]], 'quad_num_cat': [[{'key': 8, 'value': 5.0}], [{'key': 9, 'value': 5.0}], [{'key': 10, 'value': 5.0}], [{'key': 8, 'value': 6.0}], [{'key': 9, 'value': 6.0}], [{'key': 10, 'value': 6.0}], [{'key': 8, 'value': 7.0}], [{'key': 9, 'value': 7.0}], [{'key': 10, 'value': 7.0}]], 'quad_cat': [[{'key1': 8, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 9, 'value': 1.0}], [{'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 9, 'key2': 9, 'value': 1.0}], [{'key1': 9, 'key2': 10, 'value': 1.0}], [{'key1': 10, 'key2': 10, 'value': 1.0}]]}"))
    assert(res[2][0] == eval("{'N': 1, 'lin_num': [2.0, 1.0, 3.0], 'quad_num': [4.0, 2.0, 6.0, 1.0, 3.0, 9.0], 'lin_cat': [[{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 2.0}], [{'key': 6, 'value': 2.0}], [{'key': 8, 'value': 2.0}], [{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}], [{'key': 4, 'value': 3.0}], [{'key': 6, 'value': 3.0}], [{'key': 8, 'value': 3.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 1.0}]]}"))
    assert(res[3][0] == eval("{'N': 1, 'lin_num': [5.0, 7.0, 6.0], 'quad_num': [25.0, 35.0, 30.0, 49.0, 42.0, 36.0], 'lin_cat': [[{'key': 8, 'value': 1.0}], [{'key': 10, 'value': 1.0}], [{'key': 12, 'value': 1.0}]], 'quad_num_cat': [[{'key': 8, 'value': 5.0}], [{'key': 10, 'value': 5.0}], [{'key': 12, 'value': 5.0}], [{'key': 8, 'value': 7.0}], [{'key': 10, 'value': 7.0}], [{'key': 12, 'value': 7.0}], [{'key': 8, 'value': 6.0}], [{'key': 10, 'value': 6.0}], [{'key': 12, 'value': 6.0}]], 'quad_cat': [[{'key1': 8, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 8, 'key2': 12, 'value': 1.0}], [{'key1': 10, 'key2': 10, 'value': 1.0}], [{'key1': 10, 'key2': 12, 'value': 1.0}], [{'key1': 12, 'key2': 12, 'value': 1.0}]]}"))
    assert(res[4][0] == eval("{'N': 1, 'lin_num': [2.0, 1.0, 3.0], 'quad_num': [4.0, 2.0, 6.0, 1.0, 3.0, 9.0], 'lin_cat': [[{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 2.0}], [{'key': 6, 'value': 2.0}], [{'key': 8, 'value': 2.0}], [{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}], [{'key': 4, 'value': 3.0}], [{'key': 6, 'value': 3.0}], [{'key': 8, 'value': 3.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 1.0}]]}"))



def test_lift_single_numerical(duckdb_conn):
    duckdb_conn.execute("SELECT to_cofactor(e) from test")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 1, 'lin_num': [], 'quad_num': [], 'lin_cat': [[{'key': 5, 'value': 1.0}]], 'quad_num_cat': [], 'quad_cat': [[{'key1': 5, 'key2': 5, 'value': 1.0}]]}"))
    assert(res[1][0] == eval("{'N': 1, 'lin_num': [], 'quad_num': [], 'lin_cat': [[{'key': 9, 'value': 1.0}]], 'quad_num_cat': [], 'quad_cat': [[{'key1': 9, 'key2': 9, 'value': 1.0}]]}"))
    assert(res[2][0] == eval("{'N': 1, 'lin_num': [], 'quad_num': [], 'lin_cat': [[{'key': 6, 'value': 1.0}]], 'quad_num_cat': [], 'quad_cat': [[{'key1': 6, 'key2': 6, 'value': 1.0}]]}"))
    assert(res[3][0] == eval("{'N': 1, 'lin_num': [], 'quad_num': [], 'lin_cat': [[{'key': 10, 'value': 1.0}]], 'quad_num_cat': [], 'quad_cat': [[{'key1': 10, 'key2': 10, 'value': 1.0}]]}"))
    assert(res[4][0] == eval("{'N': 1, 'lin_num': [], 'quad_num': [], 'lin_cat': [[{'key': 6, 'value': 1.0}]], 'quad_num_cat': [], 'quad_cat': [[{'key1': 6, 'key2': 6, 'value': 1.0}]]}"))

def test_lift_single_categorical (duckdb_conn):
    duckdb_conn.execute("SELECT to_cofactor(a) from test")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 1, 'lin_num': [1.0], 'quad_num': [1.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"))
    assert(res[1][0] == eval("{'N': 1, 'lin_num': [5.0], 'quad_num': [25.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"))
    assert(res[2][0] == eval("{'N': 1, 'lin_num': [2.0], 'quad_num': [4.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"))
    assert(res[3][0] == eval("{'N': 1, 'lin_num': [5.0], 'quad_num': [25.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"))
    assert(res[4][0] == eval("{'N': 1, 'lin_num': [2.0], 'quad_num': [4.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"))

def test_lift_with_where(duckdb_conn):
    duckdb_conn.execute("SELECT to_cofactor(a,b,c,d,e,f) from test where gb = 2")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 1, 'lin_num': [2.0, 1.0, 3.0], 'quad_num': [4.0, 2.0, 6.0, 1.0, 3.0, 9.0], 'lin_cat': [[{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 2.0}], [{'key': 6, 'value': 2.0}], [{'key': 8, 'value': 2.0}], [{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}], [{'key': 4, 'value': 3.0}], [{'key': 6, 'value': 3.0}], [{'key': 8, 'value': 3.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 1.0}]]}"))
    assert(res[1][0] == eval("{'N': 1, 'lin_num': [2.0, 1.0, 3.0], 'quad_num': [4.0, 2.0, 6.0, 1.0, 3.0, 9.0], 'lin_cat': [[{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 2.0}], [{'key': 6, 'value': 2.0}], [{'key': 8, 'value': 2.0}], [{'key': 4, 'value': 1.0}], [{'key': 6, 'value': 1.0}], [{'key': 8, 'value': 1.0}], [{'key': 4, 'value': 3.0}], [{'key': 6, 'value': 3.0}], [{'key': 8, 'value': 3.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 1.0}]]}"))


def test_lift_with_sum_of_columns (duckdb_conn):
    duckdb_conn.execute("SELECT to_cofactor(a+b+c) from test")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 1, 'lin_num': [6.0], 'quad_num': [36.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"));
    assert(res[1][0] == eval("{'N': 1, 'lin_num': [18.0], 'quad_num': [324.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"));
    assert(res[2][0] == eval("{'N': 1, 'lin_num': [6.0], 'quad_num': [36.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}"));

