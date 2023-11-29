

#ifndef TEST_LDA_IMPUTE_H
#define TEST_LDA_IMPUTE_H

#include <duckdb.hpp>

void LDA_impute(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);
duckdb::unique_ptr<duckdb::FunctionData>
LDA_impute_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);
duckdb::unique_ptr<duckdb::BaseStatistics>
LDA_impute_stats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input);

#endif //TEST_LDA_IMPUTE_H
