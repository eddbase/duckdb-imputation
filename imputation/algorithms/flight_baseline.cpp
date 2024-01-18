
#include "flight_baseline.h"
#include <iostream>
#include "partition.h"
#include <iterator>
void run_flight_baseline(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters, const std::string sort){

    build_list_of_uniq_categoricals(cat_columns, con, table_name);
    auto begin = std::chrono::high_resolution_clock::now();

    init_baseline(table_name, con_columns_nulls, cat_columns_nulls, con);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Time prepare dataset (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    std::clog<<"Time prepare dataset (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    //start MICE
    //int parallelism = con.Query("SELECT current_setting('threads')")->GetValue<int>(0, 0);
    
    for (int mice_iter =0; mice_iter<mice_iters;mice_iter++){
        //continuous cols
        for(auto &col_null : con_columns_nulls) {
            std::cout<<"\n\nColumn: "<<col_null<<"\n\n";
            //select
            std::string delta_query = "SELECT triple_sum_no_lift(";
            for (auto &col: con_columns)
                delta_query += col + ", ";
            for (auto &col: cat_columns)
                delta_query += col + ", ";

            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " ) FROM " + table_name + "_complete WHERE "+col_null+"_IS_NULL IS FALSE";

            begin = std::chrono::high_resolution_clock::now();
            duckdb::Value train_triple = con.Query(delta_query)->GetValue(0,0);
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            //train
            auto it = std::find(con_columns.begin(), con_columns.end(), col_null);
            size_t label_index = it - con_columns.begin();
            std::cout<<"Label index "<<label_index<<"\n";

            begin = std::chrono::high_resolution_clock::now();
            std::vector <double> params = Triple::ridge_linear_regression(train_triple, label_index, 0.001, 0, 1000, true);
            end = std::chrono::high_resolution_clock::now();
            std::clog<<"Train time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            //predict query
            std::string new_val = std::to_string((float) params[0])+" + (";
            for(size_t i=0; i< con_columns.size(); i++) {
                if (i==label_index)
                    continue;
                new_val+="("+ std::to_string((float)params[i+1])+" * "+con_columns[i]+")+";
            }

            std::string cat_columns_query;
            query_categorical_num(cat_columns, cat_columns_query, std::vector<float>(params.begin()+con_columns.size(), params.end()-1));
            new_val+=cat_columns_query+" + (sqrt(-2 * ln(random()))*cos(2*pi()*random()) *"+ std::to_string(params[params.size()-1])+"))::FLOAT";
                //update
            std::cout<<"CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+new_val+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete\n";

            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+new_val+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
        }
        std::cout<<"Starting categorical variables..."<<std::endl;
        for(auto &col_null : cat_columns_nulls) {
            std::string delta_query = "SELECT triple_sum_no_lift(";
            for (auto &col: con_columns)
                delta_query += col + ", ";
            for (auto &col: cat_columns)
                delta_query += col + ", ";

            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " ) FROM "+table_name+"_complete WHERE "+col_null+"_IS_NULL IS FALSE";

            //execute delta
            auto begin = std::chrono::high_resolution_clock::now();
            duckdb::Value train_triple = con.Query(delta_query)->GetValue(0,0);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            //train
            auto it = std::find(cat_columns.begin(), cat_columns.end(), col_null);
            size_t label_index = it - cat_columns.begin();
            std::cout<<"Categorical label index "<<label_index<<"\n";
            begin = std::chrono::high_resolution_clock::now();
            auto train_params =  lda_train(train_triple, label_index, 0.4);
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Train Time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Train Time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            std::string cat_columns_query;
            std::string predict_column_query;
            query_categorical(cat_columns, label_index, cat_columns_query, predict_column_query);
            //impute
            std::string delimiter = ", ";
            std::stringstream  s;
            copy(con_columns.begin(),con_columns.end(), std::ostream_iterator<std::string>(s,delimiter.c_str()));
            std::string num_cols_query = s.str();
            num_cols_query.pop_back();
            num_cols_query.pop_back();
            std::string select_stmt = (" list_extract("+predict_column_query+", predict_lda("+train_params.ToString()+"::FLOAT[], "+num_cols_query+", "+cat_columns_query+")+1)");

            //update 2 missing values
            std::cout<<"CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete\n";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

        }
    }

}