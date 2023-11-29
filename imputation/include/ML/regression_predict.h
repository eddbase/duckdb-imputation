

#ifndef DUCKDB_REGRESSION_PREDICT_H
#define DUCKDB_REGRESSION_PREDICT_H

#include <duckdb.hpp>
namespace Triple {
    duckdb::unique_ptr<duckdb::FunctionData>
    ImputeHackBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

    void ImputeHackFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);

    duckdb::unique_ptr<duckdb::BaseStatistics>
    ImputeHackStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input);
}

#endif //DUCKDB_REGRESSION_PREDICT_H
