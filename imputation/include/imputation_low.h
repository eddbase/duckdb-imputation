

#ifndef TEST_FLIGHT_PARTITION_H
#define TEST_FLIGHT_PARTITION_H

#include <duckdb.hpp>
#include <string>
void run_MICE_low(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters);


#endif //TEST_FLIGHT_PARTITION_H
