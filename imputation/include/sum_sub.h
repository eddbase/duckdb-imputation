


#ifndef DUCKDB_TRIPLE_SUB_H
#define DUCKDB_TRIPLE_SUB_H

#include <duckdb.hpp>


namespace Triple{
    duckdb::Value subtract_triple(duckdb::Value &triple_1, duckdb::Value &triple_2);
    duckdb::Value sum_triple(const duckdb::Value &triple_1, const duckdb::Value &triple_2);
    duckdb::Value sum_nb_triple(const duckdb::Value &triple_1, const duckdb::Value &triple_2);
}


#endif //DUCKDB_TRIPLE_SUB_H
