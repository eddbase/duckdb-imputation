

#ifndef TEST_TRAIN_RETAILER_H
#define TEST_TRAIN_RETAILER_H

#include <string>
namespace Retailer {
    void train_mat_sql_lift(const std::string &path, bool materialized);

    void train_mat_custom_lift(const std::string &path, bool materialized, bool categorical);

    void train_factorized(const std::string &path, bool categorical);
}

#endif //TEST_TRAIN_RETAILER_H
