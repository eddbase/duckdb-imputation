

#ifndef TEST_TRAIN_FLIGHT_H
#define TEST_TRAIN_FLIGHT_H

#include <string>
namespace Flight {
    void train_mat_sql_lift(const std::string &path, bool materialized);

    void train_mat_custom_lift(const std::string &path, bool materialized, bool categorical);

    void train_factorized(const std::string &path, bool categorical);
    void test(const std::string &path);
}

#endif //TEST_TRAIN_FLIGHT_H
