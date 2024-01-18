

#ifndef TEST_COLUMN_SCALABILITY_H
#define TEST_COLUMN_SCALABILITY_H

#include <duckdb.hpp>

void scalability_col_exp(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters, const std::vector<std::string> &assume_col_con_nulls);

#endif //TEST_COLUMN_SCALABILITY_H
