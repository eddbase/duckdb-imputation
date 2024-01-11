//
// Created by Massimo Perini on 10/01/2024.
//

#ifndef DUCKDB_REGRESSION_H
#define DUCKDB_REGRESSION_H

#include "duckdb.hpp"
namespace ML {

void ridge_linear_regression(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData>
ImputeHackBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
               duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void ImputeHackFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);

duckdb::unique_ptr<duckdb::BaseStatistics>
ImputeHackStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input);
}

#endif // DUCKDB_REGRESSION_H
