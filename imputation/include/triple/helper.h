

#ifndef TEST_HELPER_H
#define TEST_HELPER_H

#include <duckdb.hpp>

namespace Triple {

    void register_functions(duckdb::ClientContext &context, const std::vector<size_t> &n_con_columns, const std::vector<size_t> &n_cat_columns);

} // Triple

#endif //TEST_HELPER_H
