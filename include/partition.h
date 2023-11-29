

#ifndef TEST_PARTITION_H
#define TEST_PARTITION_H

#include <string>
#include <duckdb.hpp>


void partition(const std::string &table_name, const std::vector<std::string> &con_columns, const std::vector<std::string> &con_columns_nulls,
               const std::vector<std::string> &cat_columns, const std::vector<std::string> &cat_columns_nulls, duckdb::Connection &con, const std::string& order = "");
void partition_inverse(const std::string &table_name, const std::vector<std::string> &con_columns, const std::vector<std::string> &con_columns_nulls,
                       const std::vector<std::string> &cat_columns, const std::vector<std::string> &cat_columns_nulls, duckdb::Connection &con, const std::string& order = "");
void build_list_of_uniq_categoricals(const std::vector<std::string> &cat_columns, duckdb::Connection &con, const std::string &table_name);
void init_baseline(const std::string &table_name, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, duckdb::Connection &con);
void partition_reduce_col_null(const std::string &table_name, const std::vector<std::string> &con_columns, const std::vector<std::string> &con_columns_nulls,
                               const std::vector<std::string> &cat_columns, const std::vector<std::string> &cat_columns_nulls, duckdb::Connection &con, const std::vector<std::string> &assume_columns_nulls);

void query_categorical(const std::vector<std::string> &cat_columns, size_t label, std::string &cat_columns_query, std::string &predict_column_query);
void query_categorical_num(const std::vector<std::string> &cat_columns, std::string &predict_column_query, const std::vector<float> &cat_columns_parameters);
void build_list_of_uniq_categoricals(const std::vector<std::pair<std::string, std::string>> &cat_columns, duckdb::Connection &con);
void drop_partition(const std::string &table_name, const std::vector<std::string> &con_columns_nulls,
                    const std::vector<std::string> &cat_columns_nulls, duckdb::Connection &con);


#endif //TEST_PARTITION_H
