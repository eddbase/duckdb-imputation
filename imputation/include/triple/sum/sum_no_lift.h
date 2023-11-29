

#ifndef DUCKDB_SUM_NO_LIFT_H
#define DUCKDB_SUM_NO_LIFT_H


#include <vector>
#include <duckdb.hpp>

namespace Triple {

    duckdb::unique_ptr<duckdb::FunctionData>
    SumNoLiftBind(duckdb::ClientContext &context, duckdb::AggregateFunction &function,
                     duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

    void SumNoLift(duckdb::Vector inputs[], duckdb::AggregateInputData &aggr_input_data, idx_t input_count,
                            duckdb::Vector &state_vector, idx_t count);

}


#endif //DUCKDB_SUM_NO_LIFT_H
