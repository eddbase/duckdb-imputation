#include <iostream>

#include <duckdb.hpp>

#include <helper.h>
#include "partition.h"

#include <vector>
#include <ML/lda.h>


int main(){
    std::cout<<"Hello world";
    duckdb::DBConfig c;
    c.SetOption("allow_unsigned_extensions", duckdb::Value(true));
    duckdb::DuckDB db(":memory:", &c);
    duckdb::Connection con(db);

    std::vector<std::string> con_columns = {"a", "b", "c"};;
    std::vector<std::string> cat_columns = {"d", "e", "f"};
    ML_lib::register_functions(*con.context, {con_columns.size()}, {cat_columns.size()});

    std::string ttt = "CREATE TABLE test(gb INTEGER, a FLOAT, b FLOAT, c FLOAT, d INTEGER, e INTEGER, f INTEGER);";
    con.Query(ttt);
    ttt = "INSERT INTO test VALUES (1,1,2,3,4,5,6), (1,5,6,7,8,9,10), (2,2,1,3,4,6,8), (1,5,7,6,8,10,12), (2,2,1,3,4,6,8)";
    con.Query(ttt);

    std::string cat_columns_query;
    std::string predict_column_query;
    size_t label = 0;//label inside categoricals
    build_list_of_uniq_categoricals(cat_columns, con, "test");
    con.Query("SELECT * FROM test")->Print();
    con.Query("SELECT triple_sum_no_lift(a,b,c,d,e,f) FROM test")->Print();
    auto triple = con.Query("SELECT triple_sum_no_lift(a,b,c,d,e,f) FROM test")->GetValue(0,0);
    auto train_params =  lda_train(triple, label, 0.4);
    con.Query("SELECT list_extract("+predict_column_query+", predict_lda("+train_params.ToString()+"::FLOAT[], a, b,c,"+cat_columns_query+")+1) FROM test")->Print();
}