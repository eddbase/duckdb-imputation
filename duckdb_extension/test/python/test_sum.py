import duckdb
import os
import pytest

# Get a fresh connection to DuckDB with the duckdb_imputation extension binary loaded
@pytest.fixture
def duckdb_conn():
    extension_binary = os.getenv('DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect('', config={'allow_unsigned_extensions': 'true'})
    conn.execute(f"load '{extension_binary}'")

    conn.execute("CREATE TABLE test(gb INTEGER, a FLOAT, b FLOAT, c FLOAT, d INTEGER, e INTEGER, f INTEGER);")
    conn.execute("INSERT INTO test VALUES (1,1,2,3,4,5,6), (1,5,6,7,8,9,10), (2,2,1,3,4,6,8), (1,5,7,6,8,10,12), (2,2,1,3,4,6,8)")

    return conn



def test_sum_no_lift_everything(duckdb_conn):
    duckdb_conn.execute("SELECT sum_to_triple_3_3(a,b,c,d,e,f) from test");
    res = duckdb_conn.fetchall()
    assert(res[0][0] == "{'N': 5, 'lin_agg': [15.0, 17.0, 22.0], 'quad_agg': [59.0, 71.0, 80.0, 91.0, 96.0, 112.0], 'lin_cat': [[{'key': 4, 'value': 3.0}, {'key': 8, 'value': 2.0}], [{'key': 5, 'value': 1.0}, {'key': 6, 'value': 2.0}, {'key': 9, 'value': 1.0}, {'key': 10, 'value': 1.0}], [{'key': 6, 'value': 1.0}, {'key': 8, 'value': 2.0}, {'key': 10, 'value': 1.0}, {'key': 12, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 5.0}, {'key': 8, 'value': 10.0}], [{'key': 5, 'value': 1.0}, {'key': 6, 'value': 4.0}, {'key': 9, 'value': 5.0}, {'key': 10, 'value': 5.0}], [{'key': 6, 'value': 1.0}, {'key': 8, 'value': 4.0}, {'key': 10, 'value': 5.0}, {'key': 12, 'value': 5.0}], [{'key': 4, 'value': 4.0}, {'key': 8, 'value': 13.0}], [{'key': 5, 'value': 2.0}, {'key': 6, 'value': 2.0}, {'key': 9, 'value': 6.0}, {'key': 10, 'value': 7.0}], [{'key': 6, 'value': 2.0}, {'key': 8, 'value': 2.0}, {'key': 10, 'value': 6.0}, {'key': 12, 'value': 7.0}], [{'key': 4, 'value': 9.0}, {'key': 8, 'value': 13.0}], [{'key': 5, 'value': 3.0}, {'key': 6, 'value': 6.0}, {'key': 9, 'value': 7.0}, {'key': 10, 'value': 6.0}], [{'key': 6, 'value': 3.0}, {'key': 8, 'value': 6.0}, {'key': 10, 'value': 7.0}, {'key': 12, 'value': 6.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 3.0}, {'key1': 8, 'key2': 8, 'value': 2.0}], [{'key1': 4, 'key2': 5, 'value': 1.0}, {'key1': 4, 'key2': 6, 'value': 2.0}, {'key1': 8, 'key2': 9, 'value': 1.0}, {'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}, {'key1': 4, 'key2': 8, 'value': 2.0}, {'key1': 8, 'key2': 10, 'value': 1.0}, {'key1': 8, 'key2': 12, 'value': 1.0}], [{'key1': 5, 'key2': 5, 'value': 1.0}, {'key1': 6, 'key2': 6, 'value': 2.0}, {'key1': 9, 'key2': 9, 'value': 1.0}, {'key1': 10, 'key2': 10, 'value': 1.0}], [{'key1': 5, 'key2': 6, 'value': 1.0}, {'key1': 6, 'key2': 8, 'value': 2.0}, {'key1': 9, 'key2': 10, 'value': 1.0}, {'key1': 10, 'key2': 12, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}, {'key1': 8, 'key2': 8, 'value': 2.0}, {'key1': 10, 'key2': 10, 'value': 1.0}, {'key1': 12, 'key2': 12, 'value': 1.0}]]}");



def test_sum_no_lift_group_by(duckdb_conn):
    duckdb_conn.execute("SELECT sum_to_triple_3_3(a,b,c,d,e,f) from test GROUP BY gb");
    res = duckdb_conn.fetchall()
    assert(res[0][0] == "{'N': 2, 'lin_agg': [6.0, 8.0, 10.0], 'quad_agg': [26.0, 32.0, 38.0, 40.0, 48.0, 58.0], 'lin_cat': [[{'key': 4, 'value': 1.0}, {'key': 8, 'value': 1.0}], [{'key': 5, 'value': 1.0}, {'key': 9, 'value': 1.0}], [{'key': 6, 'value': 1.0}, {'key': 10, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 1.0}, {'key': 8, 'value': 5.0}], [{'key': 5, 'value': 1.0}, {'key': 9, 'value': 5.0}], [{'key': 6, 'value': 1.0}, {'key': 10, 'value': 5.0}], [{'key': 4, 'value': 2.0}, {'key': 8, 'value': 6.0}], [{'key': 5, 'value': 2.0}, {'key': 9, 'value': 6.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 6.0}], [{'key': 4, 'value': 3.0}, {'key': 8, 'value': 7.0}], [{'key': 5, 'value': 3.0}, {'key': 9, 'value': 7.0}], [{'key': 6, 'value': 3.0}, {'key': 10, 'value': 7.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 1.0}, {'key1': 8, 'key2': 8, 'value': 1.0}], [{'key1': 4, 'key2': 5, 'value': 1.0}, {'key1': 8, 'key2': 9, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 1.0}, {'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 5, 'key2': 5, 'value': 1.0}, {'key1': 9, 'key2': 9, 'value': 1.0}], [{'key1': 5, 'key2': 6, 'value': 1.0}, {'key1': 9, 'key2': 10, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 1.0}, {'key1': 10, 'key2': 10, 'value': 1.0}]]}");
    assert(res[1][0] == "{'N': 3, 'lin_agg': [9.0, 9.0, 12.0], 'quad_agg': [33.0, 39.0, 42.0, 51.0, 48.0, 54.0], 'lin_cat': [[{'key': 4, 'value': 2.0}, {'key': 8, 'value': 1.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 1.0}], [{'key': 8, 'value': 2.0}, {'key': 12, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 4.0}, {'key': 8, 'value': 5.0}], [{'key': 6, 'value': 4.0}, {'key': 10, 'value': 5.0}], [{'key': 8, 'value': 4.0}, {'key': 12, 'value': 5.0}], [{'key': 4, 'value': 2.0}, {'key': 8, 'value': 7.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 7.0}], [{'key': 8, 'value': 2.0}, {'key': 12, 'value': 7.0}], [{'key': 4, 'value': 6.0}, {'key': 8, 'value': 6.0}], [{'key': 6, 'value': 6.0}, {'key': 10, 'value': 6.0}], [{'key': 8, 'value': 6.0}, {'key': 12, 'value': 6.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 2.0}, {'key1': 8, 'key2': 8, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 2.0}, {'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 2.0}, {'key1': 8, 'key2': 12, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 2.0}, {'key1': 10, 'key2': 10, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 2.0}, {'key1': 10, 'key2': 12, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 2.0}, {'key1': 12, 'key2': 12, 'value': 1.0}]]}");

def test_sum_no_lift_having (duckdb_conn):
    duckdb_conn.execute("SELECT sum_to_triple_3_3(a,b,c,d,e,f) from test GROUP BY gb HAVING gb = 2")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == "{'N': 3, 'lin_agg': [9.0, 9.0, 12.0], 'quad_agg': [33.0, 39.0, 42.0, 51.0, 48.0, 54.0], 'lin_cat': [[{'key': 4, 'value': 2.0}, {'key': 8, 'value': 1.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 1.0}], [{'key': 8, 'value': 2.0}, {'key': 12, 'value': 1.0}]], 'quad_num_cat': [[{'key': 4, 'value': 4.0}, {'key': 8, 'value': 5.0}], [{'key': 6, 'value': 4.0}, {'key': 10, 'value': 5.0}], [{'key': 8, 'value': 4.0}, {'key': 12, 'value': 5.0}], [{'key': 4, 'value': 2.0}, {'key': 8, 'value': 7.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 7.0}], [{'key': 8, 'value': 2.0}, {'key': 12, 'value': 7.0}], [{'key': 4, 'value': 6.0}, {'key': 8, 'value': 6.0}], [{'key': 6, 'value': 6.0}, {'key': 10, 'value': 6.0}], [{'key': 8, 'value': 6.0}, {'key': 12, 'value': 6.0}]], 'quad_cat': [[{'key1': 4, 'key2': 4, 'value': 2.0}, {'key1': 8, 'key2': 8, 'value': 1.0}], [{'key1': 4, 'key2': 6, 'value': 2.0}, {'key1': 8, 'key2': 10, 'value': 1.0}], [{'key1': 4, 'key2': 8, 'value': 2.0}, {'key1': 8, 'key2': 12, 'value': 1.0}], [{'key1': 6, 'key2': 6, 'value': 2.0}, {'key1': 10, 'key2': 10, 'value': 1.0}], [{'key1': 6, 'key2': 8, 'value': 2.0}, {'key1': 10, 'key2': 12, 'value': 1.0}], [{'key1': 8, 'key2': 8, 'value': 2.0}, {'key1': 12, 'key2': 12, 'value': 1.0}]]}");

def test_sum_group_by(duckdb_conn):
    duckdb_conn.execute("SELECT sum_to_triple_3_3(a,b,c,d,e,f) from test GROUP BY gb")
    res = duckdb_conn.fetchall()
    duckdb_conn.execute("SELECT sum_triple(to_cofactor(a,b,c,d,e,f)) from test GROUP BY gb")
    res2 = duckdb_conn.fetchall()
    assert (res[0][0] == res2[0][0])

def test_sum_having(duckdb_conn):
    duckdb_conn.execute("SELECT sum_to_triple_3_3(a,b,c,d,e,f) from test GROUP BY gb HAVING gb = 2")
    res = duckdb_conn.fetchall()
    duckdb_conn.execute("SELECT sum_triple(to_cofactor(a,b,c,d,e,f)) from test GROUP BY gb HAVING gb = 2")
    res2 = duckdb_conn.fetchall()
    assert (res[0][0] == res2[0][0])


