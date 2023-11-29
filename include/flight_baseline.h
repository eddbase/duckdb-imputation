
#ifndef TEST_FLIGHT_BASELINE_H
#define TEST_FLIGHT_BASELINE_H

#include <duckdb.hpp>
#include <string>

void run_flight_baseline(duckdb::Connection &con, const std::vector<std::string> &con_columns, const std::vector<std::string> &cat_columns, const std::vector<std::string> &con_columns_nulls, const std::vector<std::string> &cat_columns_nulls, const std::string &table_name, size_t mice_iters, const std::string sort);

#endif //TEST_FLIGHT_BASELINE_H
