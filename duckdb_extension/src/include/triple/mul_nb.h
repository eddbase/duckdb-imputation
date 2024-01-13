//
// Created by Massimo Perini on 12/01/2024.
//

#ifndef DUCKDB_MUL_NB_H
#define DUCKDB_MUL_NB_H

namespace Triple{

void multiply_nb(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result);
duckdb::unique_ptr<duckdb::FunctionData>
multiply_nb_bind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                 duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments);


}


#endif // DUCKDB_MUL_NB_H
