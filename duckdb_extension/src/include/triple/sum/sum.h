

#ifndef DUCKDB_TRIPLE_SUM_H
#define DUCKDB_TRIPLE_SUM_H

#include <vector>
#include <duckdb.hpp>
#include <map>
#include <unordered_map>
#include <iostream>

//#include <boost/unordered/unordered_flat_map.hpp>

namespace Triple {

duckdb::unique_ptr<duckdb::FunctionData>
SumBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

void Sum(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
         duckdb::Vector &state_vector, idx_t count);


duckdb::Value sum_triple(const duckdb::Value &triple_1, const duckdb::Value &triple_2);

}

#endif //DUCKDB_TRIPLE_SUM_H
