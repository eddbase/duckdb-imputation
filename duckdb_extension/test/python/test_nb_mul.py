import duckdb
import os
import pytest

# Get a fresh connection to DuckDB with the duckdb_imputation extension binary loaded
@pytest.fixture
def duckdb_conn():
    extension_binary = "../../build/release/extension/duckdb_imputation/duckdb_imputation.duckdb_extension"#os.getenv('DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `DUCKDB_IMPUTATION_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect(':memory:', config={'allow_unsigned_extensions' : 'true'})
    conn.execute(f"INSTALL '{extension_binary}'")
    conn.execute(f"load duckdb_imputation")

    conn.execute("CREATE TABLE test(gb INTEGER, a FLOAT, b FLOAT, c FLOAT, d INTEGER, e INTEGER, f INTEGER);")
    conn.execute("INSERT INTO test VALUES (1,1,2,3,4,5,6), (1,5,6,7,8,9,10), (2,2,1,3,4,6,8), (2,5,7,6,8,10,12), (2,2,1,3,4,6,8)")

    return conn



def test_sum_no_lift_everything(duckdb_conn):
    duckdb_conn.execute("SELECT multiply_nb_agg(A, B) FROM ((SELECT sum_to_nb_agg_2_2(b,c,d,e) AS A FROM test where gb = 1) INNER JOIN "
                        "(SELECT sum_to_nb_agg_2_2(a,c,d,f) AS B FROM test where gb = 2) ON TRUE)")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 6, 'lin_num': [24.0, 30.0, 18.0, 24.0], 'quad_num': [120.0, 174.0, 66.0, 108.0], 'lin_cat': [[{'key': 4, 'value': 3.0}, {'key': 8, 'value': 3.0}], [{'key': 5, 'value': 3.0}, {'key': 9, 'value': 3.0}], [{'key': 4, 'value': 4.0}, {'key': 8, 'value': 2.0}], [{'key': 8, 'value': 4.0}, {'key': 12, 'value': 2.0}]]}"))



def test_sum_no_lift_groupby(duckdb_conn):
    duckdb_conn.execute("SELECT multiply_nb_agg(A, B) FROM ((SELECT sum_to_nb_agg_2_2(b,c,d,e) AS A FROM test GROUP BY gb) INNER JOIN "
                        "(SELECT sum_to_nb_agg_2_2(a,c,d,f) AS B FROM test GROUP BY gb) ON TRUE)")
    res = duckdb_conn.fetchall()
    assert(res[0][0] == eval("{'N': 4, 'lin_num': [16.0, 20.0, 12.0, 20.0], 'quad_num': [80.0, 116.0, 52.0, 116.0], 'lin_cat': [[{'key': 4, 'value': 2.0}, {'key': 8, 'value': 2.0}], [{'key': 5, 'value': 2.0}, {'key': 9, 'value': 2.0}], [{'key': 4, 'value': 2.0}, {'key': 8, 'value': 2.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 2.0}]]}"))
    assert(res[1][0] == eval("{'N': 6, 'lin_num': [18.0, 24.0, 27.0, 36.0], 'quad_num': [102.0, 108.0, 99.0, 162.0], 'lin_cat': [[{'key': 4, 'value': 4.0}, {'key': 8, 'value': 2.0}], [{'key': 6, 'value': 4.0}, {'key': 10, 'value': 2.0}], [{'key': 4, 'value': 6.0}, {'key': 8, 'value': 3.0}], [{'key': 8, 'value': 6.0}, {'key': 12, 'value': 3.0}]]}"))
    assert(res[2][0] == eval("{'N': 6, 'lin_num': [24.0, 30.0, 12.0, 20.0], 'quad_num': [120.0, 174.0, 52.0, 116.0], 'lin_cat': [[{'key': 4, 'value': 3.0}, {'key': 8, 'value': 3.0}], [{'key': 5, 'value': 3.0}, {'key': 9, 'value': 3.0}], [{'key': 4, 'value': 2.0}, {'key': 8, 'value': 2.0}], [{'key': 6, 'value': 2.0}, {'key': 10, 'value': 2.0}]]}"))
    assert(res[3][0] == eval("{'N': 9, 'lin_num': [27.0, 36.0, 27.0, 36.0], 'quad_num': [153.0, 162.0, 99.0, 162.0], 'lin_cat': [[{'key': 4, 'value': 6.0}, {'key': 8, 'value': 3.0}], [{'key': 6, 'value': 6.0}, {'key': 10, 'value': 3.0}], [{'key': 4, 'value': 6.0}, {'key': 8, 'value': 3.0}], [{'key': 8, 'value': 6.0}, {'key': 12, 'value': 3.0}]]}"))


