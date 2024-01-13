//
// Created by Massimo Perini on 12/01/2024.
//

#ifndef DUCKDB_LIFT_TO_NB_AGG_H
#define DUCKDB_LIFT_TO_NB_AGG_H

namespace Triple {
void to_nb_lift(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                duckdb::Vector &result);

duckdb::unique_ptr<duckdb::FunctionData>
to_nb_lift_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);

}

#endif // DUCKDB_LIFT_TO_NB_AGG_H
