
#include <duckdb.hpp>
#include <triple/helper.h>
#include <iostream>

#include <vector>
#include <ML/lda.h>

using namespace std;
int main(int argc, char* argv[]){

    duckdb::DuckDB db(":memory:");
    duckdb::Connection con(db);

    std::vector<std::string> con_columns = {"a", "b", "c"};;
    std::vector<std::string> cat_columns = {"d", "e", "f"};
    Triple::register_functions(*con.context, {con_columns.size()}, {cat_columns.size()});

    std::string ttt = "CREATE TABLE test(gb INTEGER, a FLOAT, b FLOAT, c FLOAT, d INTEGER, e INTEGER, f INTEGER);";
    con.Query(ttt);
    ttt = "INSERT INTO test VALUES (1,1,2,3,4,5,6), (1,5,6,7,8,9,10), (2,2,1,3,4,6,8), (1,5,7,6,8,10,12), (2,2,1,3,4,6,8)";
    con.Query(ttt);

        std::string cat_columns_query;
        std::string predict_column_query;
        size_t label = 0;//label inside categoricals
    
        build_list_of_uniq_categoricals(cat_columns, con, "test");
        query_categorical(cat_columns, label, cat_columns_query, predict_column_query);
    
        auto triple = con.Query("SELECT triple_sum_no_lift(a,b,c,d,e,f) FROM test")->GetValue(0,0);
        auto train_params =  lda_train(triple, label, 0.4);
        auto res = con.Query("SELECT list_extract([4,8], predict_lda("+train_params.ToString()+"::FLOAT[], a, b,c,list_position([5,6,9,10], e)-1, list_position([6,8,10,12], f)-1 )+1) FROM test");
        //(res->GetValue(0,0).GetValue<int>() == 4);
        //(res->GetValue(0,1).GetValue<int>() == 8);
        //(res->GetValue(0,2).GetValue<int>() == 4);
        //(res->GetValue(0,3).GetValue<int>() == 8);
        //(res->GetValue(0,4).GetValue<int>() == 4);
}
