

#ifndef DUCKDB_CUSTOM_LIFT_H
#define DUCKDB_CUSTOM_LIFT_H

#include <duckdb.hpp>


namespace Triple{
    std::string lift(std::vector<std::string> attributes);

    void CustomLift(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);
    duckdb::unique_ptr<duckdb::FunctionData> CustomLiftBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                                                              duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);
    duckdb::unique_ptr<duckdb::BaseStatistics> CustomLiftStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input);


}

#endif //DUCKDB_CUSTOM_LIFT_H
