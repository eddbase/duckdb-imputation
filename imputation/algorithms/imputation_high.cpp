

#include <imputation_low.h>
#include <iostream>
#include <partition.h>
#include <sum_sub.h>
#include <iterator>
void run_MICE_high(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters){
    //con.Query("SELECT OP_CARRIER, COUNT(*) FROM join_table WHERE WHEELS_ON_HOUR is not null GROUP BY OP_CARRIER")->Print();
    //int parallelism = con.Query("SELECT current_setting('threads')")->GetValue<int>(0, 0);
    build_list_of_uniq_categoricals(cat_columns, con, table_name);
    //partition according to n. missing values
    auto begin = std::chrono::high_resolution_clock::now();
    partition_inverse(table_name, con_columns, con_columns_nulls, cat_columns, cat_columns_nulls, con);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Time partitioning (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    std::clog<<"Time partitioning (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    //run MICE
    //compute main cofactor
    std::string query = "SELECT triple_sum_no_lift(";
    for(auto &col: con_columns){
        query +=col+", ";
    }
    for(auto &col: cat_columns){
        query +=col+", ";
    }
    query.pop_back();
    query.pop_back();
    query += " ) FROM "+table_name+"_complete_3 ";

    begin = std::chrono::high_resolution_clock::now();
    duckdb::Value static_triple = con.Query(query)->GetValue(0,0);
    end = std::chrono::high_resolution_clock::now();
    std::clog<<"Time static cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
    std::cout<<"Time static cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

    //start MICE

    for (int mice_iter =0; mice_iter<mice_iters;mice_iter++){
        //continuous cols
        for(auto &col_null : con_columns_nulls) {
            std::cout<<"\n\nColumn: "<<col_null<<"\n\n";
            //remove nulls
            std::string delta_query = "SELECT triple_sum_no_lift(";
            for (auto &col: con_columns)
                delta_query += col + ", ";
            for (auto &col: cat_columns)
                delta_query += col + ", ";

            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " ) FROM (SELECT * FROM " + table_name + "_complete_"+col_null+" UNION ALL (SELECT ";
            for(auto &col: con_columns)
                delta_query +=col+", ";
            for(auto &col: cat_columns)
                delta_query +=col+", ";
            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " FROM "+table_name+"_complete_2 WHERE "+col_null+"_IS_NOT_NULL))";
            //con.Query("SELECT COUNT(*) FROM join_table_complete_WHEELS_ON_HOUR")->Print();
            //con.Query("SELECT COUNT(*) FROM join_table_complete_2 WHERE WHEELS_ON_HOUR_IS_NULL")->Print();

            begin = std::chrono::high_resolution_clock::now();
            std::cout<<delta_query;
            duckdb::Value null_triple = con.Query(delta_query)->GetValue(0,0);
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            duckdb::Value train_triple = Triple::sum_triple(static_triple, null_triple);

            //train
            auto it = std::find(con_columns.begin(), con_columns.end(), col_null);
            size_t label_index = it - con_columns.begin();
            std::cout<<"Label index "<<label_index<<"\n";

            begin = std::chrono::high_resolution_clock::now();
            std::vector <double> params = std::vector<double>();//Triple::ridge_linear_regression(train_triple, label_index, 0.001, 0, 1000, true);
            end = std::chrono::high_resolution_clock::now();
            std::clog<<"Train time (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            //std::vector <double> params(test.size(), 0);
            //params[0] = 1;

            //predict query
            std::string new_val = std::to_string((float) params[0])+" + (";
            for(size_t i=0; i< con_columns.size(); i++) {
                if (i==label_index)
                    continue;
                new_val+="("+ std::to_string((float)params[i+1])+" * "+con_columns[i]+")+";
            }
            if(cat_columns.empty())
                new_val.pop_back();

            std::string cat_columns_query;
            query_categorical_num(cat_columns, cat_columns_query,
                                  std::vector<float>(params.begin() + con_columns.size(), params.end() - 1));
            new_val += cat_columns_query + "+(sqrt(-2 * ln(random()))*cos(2*pi()*random()) *" +
                       std::to_string(params[params.size() - 1]) + "))::FLOAT";


            //update =1 existing value in other columns
            for(auto &col_null2 : con_columns_nulls) {
                if (col_null2 == col_null)
                    continue;

                std::cout
                        << "CREATE TABLE rep AS SELECT " + new_val + " AS new_vals FROM " + table_name + "_complete_" +
                           col_null2 << "\n";
                begin = std::chrono::high_resolution_clock::now();
                con.Query("CREATE TABLE rep AS SELECT " + new_val + " AS new_vals FROM " + table_name + "_complete_" +
                          col_null2);

                //con.Query("SET threads TO 1;");
                //con.Query("CREATE TABLE rep AS SELECT new_vals from rep2");
                //con.Query("SET threads TO "+ std::to_string(parallelism));
                //con.Query("DROP TABLE rep2");

                con.Query("ALTER TABLE " + table_name + "_complete_" + col_null2 + " ALTER COLUMN " + col_null +
                          " SET DEFAULT 10;")->Print();//not adding b, replace s with rep
                end = std::chrono::high_resolution_clock::now();
                std::cout << "Time updating ==1 partition (ms): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
                std::clog<< "Time updating ==1 partition (ms): "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";

            }

            for(auto &col_null2 : cat_columns_nulls) {
                //update 1 existing value in other columns
                std::cout
                        << "CREATE TABLE rep AS SELECT " + new_val + " AS new_vals FROM " + table_name + "_complete_" +
                           col_null2 << "\n";
                begin = std::chrono::high_resolution_clock::now();
                con.Query("CREATE TABLE rep AS SELECT " + new_val + " AS new_vals FROM " + table_name + "_complete_" +
                          col_null2);

                //con.Query("SET threads TO 1;");
                //con.Query("CREATE TABLE rep AS SELECT new_vals from rep2");
                //con.Query("SET threads TO "+ std::to_string(parallelism));
                //con.Query("DROP TABLE rep2");

                con.Query("ALTER TABLE " + table_name + "_complete_" + col_null2 + " ALTER COLUMN " + col_null +
                          " SET DEFAULT 10;")->Print();//not adding b, replace s with rep
                end = std::chrono::high_resolution_clock::now();
                std::cout << "Time updating ==1 partition (ms): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
                std::clog<< "Time updating ==1 partition (ms): "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
            }

            //update 2 missing values
            std::cout<<"CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NOT_NULL IS FALSE THEN "+new_val+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete_2";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NOT_NULL IS FALSE THEN "+new_val+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete_2");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete_2 ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating >=2 partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating >=2 partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            std::cout<<"CREATE TABLE rep AS SELECT "+new_val+" AS test FROM "+table_name+"_complete_0";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT "+new_val+" AS test FROM "+table_name+"_complete_0");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete_0 ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating all missing partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating all missing partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
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
            delta_query += " ) FROM (SELECT * FROM " + table_name + "_complete_"+col_null+" UNION ALL (SELECT ";
            for(auto &col: con_columns)
                delta_query +=col+", ";
            for(auto &col: cat_columns)
                delta_query +=col+", ";
            delta_query.pop_back();
            delta_query.pop_back();
            delta_query += " FROM "+table_name+"_complete_2 WHERE "+col_null+"_IS_NOT_NULL))";

            //execute delta
            begin = std::chrono::high_resolution_clock::now();
            duckdb::Value null_triple = con.Query(delta_query)->GetValue(0,0);
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time cofactor (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            duckdb::Value train_triple = Triple::sum_triple(static_triple, null_triple);

            //train
            auto it = std::find(cat_columns.begin(), cat_columns.end(), col_null);
            size_t label_index = it - cat_columns.begin();
            std::cout<<"Categorical label index "<<label_index<<"\n";
            begin = std::chrono::high_resolution_clock::now();
            auto train_params = duckdb::Value(0);// std::vector<double>();//lda_train(train_triple, label_index, 0.4);
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

            //predict query
            //update 1 missing value

            for(auto &col_null2 : con_columns_nulls) {
                std::cout
                        << "CREATE TABLE rep AS SELECT " + select_stmt + " AS new_vals FROM " + table_name + "_complete_" +
                           col_null2 << "\n";
                begin = std::chrono::high_resolution_clock::now();
                con.Query("CREATE TABLE rep AS SELECT " + select_stmt + " AS new_vals FROM " + table_name + "_complete_" +
                          col_null2);

                //con.Query("SET threads TO 1;");
                //con.Query("CREATE TABLE rep AS SELECT new_vals from rep2");
                //con.Query("SET threads TO "+ std::to_string(parallelism));
                //con.Query("DROP TABLE rep2");

                con.Query("ALTER TABLE " + table_name + "_complete_" + col_null2 + " ALTER COLUMN " + col_null +
                          " SET DEFAULT 10;")->Print();//not adding b, replace s with rep
                end = std::chrono::high_resolution_clock::now();
                std::cout << "Time updating ==1 partition (ms): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
                std::clog<< "Time updating ==1 partition (ms): "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";

            }

            for(auto &col_null2 : cat_columns_nulls) {
                if (col_null2 == col_null)
                    continue;
                //update 1 existing value in other columns
                std::cout
                        << "CREATE TABLE rep AS SELECT " + select_stmt + " AS new_vals FROM " + table_name + "_complete_" +
                           col_null2 << "\n";
                begin = std::chrono::high_resolution_clock::now();
                con.Query("CREATE TABLE rep AS SELECT " + select_stmt + " AS new_vals FROM " + table_name + "_complete_" +
                          col_null2)->Print();

                //con.Query("SET threads TO 1;");
                //con.Query("CREATE TABLE rep AS SELECT new_vals from rep2")->Print();
                //con.Query("SET threads TO "+ std::to_string(parallelism));
                //con.Query("DROP TABLE rep2");

                con.Query("ALTER TABLE " + table_name + "_complete_" + col_null2 + " ALTER COLUMN " + col_null +
                          " SET DEFAULT 10;")->Print();//not adding b, replace s with rep
                end = std::chrono::high_resolution_clock::now();
                std::cout << "Time updating ==1 partition (ms): "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
                std::clog<< "Time updating ==1 partition (ms): "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n";
            }

            //update 2 missing values
            std::cout<<"CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NOT_NULL IS FALSE THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete_2";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT CASE WHEN "+col_null+"_IS_NOT_NULL IS FALSE THEN "+select_stmt+" ELSE "+col_null+" END AS test FROM "+table_name+"_complete_2");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete_2 ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating >=2 partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating >=2 partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";

            std::cout<<"CREATE TABLE rep AS SELECT "+select_stmt+" AS test FROM "+table_name+"_complete_0";
            begin = std::chrono::high_resolution_clock::now();
            con.Query("CREATE TABLE rep AS SELECT "+select_stmt+" AS test FROM "+table_name+"_complete_0");

            //con.Query("SET threads TO 1;");
            //con.Query("CREATE TABLE rep AS SELECT test from rep2");
            //con.Query("SET threads TO "+ std::to_string(parallelism));
            //con.Query("DROP TABLE rep2");

            con.Query("ALTER TABLE "+table_name+"_complete_0 ALTER COLUMN "+col_null+" SET DEFAULT 10;")->Print();//not adding b, replace s with rep
            end = std::chrono::high_resolution_clock::now();
            std::cout<<"Time updating all missing partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
            std::clog<<"Time updating all missing partition (ms): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()<<"\n";
        }
    }
    drop_partition(table_name, con_columns_nulls, cat_columns_nulls, con);
}