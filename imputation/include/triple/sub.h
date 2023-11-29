


#ifndef DUCKDB_TRIPLE_SUB_H
#define DUCKDB_TRIPLE_SUB_H

#include <duckdb.hpp>


namespace Triple{
    void TripleSubFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);
    duckdb::unique_ptr<duckdb::FunctionData> TripleSubBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                                                          duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);
    duckdb::unique_ptr<duckdb::BaseStatistics> TripleSubStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input);
    duckdb::Value subtract_triple(duckdb::Value &triple_1, duckdb::Value &triple_2);
    }


#endif //DUCKDB_TRIPLE_SUB_H
