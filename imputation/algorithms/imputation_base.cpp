
#include <imputation_baseline.h>
#include <iostream>
#include <partition.h>
#include <iterator>
void run_MICE_baseline(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters){

    auto begin = std::chrono::high_resolution_clock::now();
    init_baseline(table_name, con_columns_nulls, cat_columns_nulls, con);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Time prepare dataset (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    std::clog<<"Time prepare dataset (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    //start MICE
    //int parallelism = con.Query("SELECT current_setting('threads')")->GetValue<int>(0, 0);

    for (int mice_iter =0; mice_iter<mice_iters;mice_iter++){
        //continuous cols
        std::cout<<"Starting categorical variables..."<<std::endl;
        for(auto &col_null : cat_columns_nulls) {

            std::string delta_query = "SELECT sum_to_triple_"+std::to_string(con_columns.size())+"_"+std::to_string(cat_columns.size())+"(";
            for (auto &col: con_columns)
                delta_query += col + ", ";
            for (auto &col: cat_columns)
                delta_query += col + ", ";

            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " ) FROM "+table_name+"_complete WHERE "+col_null+"_IS_NULL IS FALSE";

            //execute delta
            auto begin = std::chrono::high_resolution_clock::now();
            con.Query(delta_query)->Print();
            duckdb::Value train_triple = con.Query(delta_query)->GetValue(0,0);
            //(label from 0, shrinkage :float, normalize :boolean)
            auto end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            //train
            auto it = std::find(cat_columns.begin(), cat_columns.end(), col_null);
            size_t label_index = it - cat_columns.begin();
            std::cout<<"Categorical label index "<<label_index<<"\n";
            begin = std::chrono::high_resolution_clock::now();

            con.Query("select lda_train("+train_triple.ToString()+"::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][]), "+ std::to_string(label_index)+", 0.001::FLOAT, false)")->Print();

            //lda_train (triple: triple, label from 0, shrinkage :float, normalize :boolean)
            std::string train_params = con.Query("select lda_train("+train_triple.ToString()+"::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][]), "+ std::to_string(label_index)+", 0.001::FLOAT, false)")->GetValue(0,0).ToString();

            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Train Time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Train Time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            std::string cat_columns_query;
            std::string predict_column_query;
            //impute
            std::string delimiter = ", ";
            std::stringstream  s;
            copy(con_columns.begin(),con_columns.end(), std::ostream_iterator<std::string>(s,delimiter.c_str()));

            std::string columns_without_missing;
            for (auto &col: con_columns) {
                columns_without_missing += col + ", ";
            }
            for (auto &col: cat_columns) {
                if (col == col_null)
                    continue;
                columns_without_missing += col + ", ";
            }
            columns_without_missing.pop_back();
            columns_without_missing.pop_back();


            std::string select_stmt = ("lda_predict("+train_params+"::FLOAT[], false, "+columns_without_missing+")");

            std::cout<<"CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete\n";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete")->Print();

            con.Query("SELECT * FROM "+table_name+"_complete")->Print();

            con.Query("ALTER TABLE "+table_name+"_complete ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
        }

        for(auto &col_null : con_columns_nulls) {
            std::cout<<"\n\nColumn: "<<col_null<<"\n\n";
            //select
            std::string delta_query = "SELECT sum_to_triple_"+std::to_string(con_columns.size())+"_"+std::to_string(cat_columns.size())+"(";
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

            //linreg_train (triple: triple, label: integer from 1, learning_rate: float, regularization: float, max_iterations: integer, include_variance: boolean, normalize: boolean)
            std::string params = con.Query("select linreg_train("+train_triple.ToString()+"::STRUCT(N int, lin_agg FLOAT[], quad_agg FLOAT[], lin_cat STRUCT(key INT, value FLOAT)[][], quad_num_cat STRUCT(key INT, value FLOAT)[][], quad_cat STRUCT(key1 INT, key2 INT, value FLOAT)[][]), "+ std::to_string(label_index)+", 0.001::FLOAT, 0::FLOAT, 10000::INTEGER, true, false)")->GetValue(0,0).ToString();
            end = std::chrono::high_resolution_clock::now();
            std::clog<<"Train time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            //predict query
            std::string columns_without_missing;
            for (auto &col: con_columns) {
                if (col == col_null)
                    continue;
                columns_without_missing += col + ", ";
            }
            for (auto &col: cat_columns)
                columns_without_missing += col + ", ";
            columns_without_missing.pop_back();
            columns_without_missing.pop_back();


            std::string predict_query = "linreg_predict("+params+"::FLOAT[], true, false, "+columns_without_missing+")";
            std::cout<<predict_query;

            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NULL THEN "+predict_query+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete")->Print();

            con.Query("ALTER TABLE "+table_name+"_complete ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating table (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
        }
    }

}