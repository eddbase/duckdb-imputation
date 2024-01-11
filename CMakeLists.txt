cmake_minimum_required(VERSION 3.1)
project(DuckDB_imputation)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    panic()
endif()

include(ExternalProject)

set(ENABLE_DOCTESTS ON)

ExternalProject_Add(
        duckdb_ep
        PREFIX "${CMAKE_BINARY_DIR}/lib/duckdb"
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/lib/duckdb"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/lib/duckdb/install"
        CMAKE_ARGS
        -G${CMAKE_GENERATOR}
        -DCMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=-std=c++17
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/lib/duckdb/install
        -DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DLEAN=ON
        DOWNLOAD_COMMAND ""
        UPDATE_COMMAND ""
        BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -j${CMAKE_BUILD_PARALLEL_LEVEL} duckdb_static duckdb
        BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/libduckdb_static.a
)

ExternalProject_Get_Property(duckdb_ep INSTALL_DIR)
set(DUCKDB_LIBRARY_PATH "${INSTALL_DIR}/lib/libduckdb_static.a")
set(DUCKDB_INCLUDE_DIR "${INSTALL_DIR}/include")
file(MAKE_DIRECTORY ${DUCKDB_INCLUDE_DIR})

#add_library(duckdb STATIC IMPORTED)
add_library(duckdb SHARED IMPORTED)
set_property(TARGET duckdb PROPERTY IMPORTED_LOCATION ${DUCKDB_LIBRARY_PATH})
set_property(TARGET duckdb APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DUCKDB_INCLUDE_DIR})
#target_link_libraries(duckdb
#        INTERFACE "${INSTALL_DIR}/lib/libduckdb_static.a"
#        )
#set_target_properties(duckdb PROPERTIES IMPORTED_LOCATION ${DUCKDB_LIB_PATH})


if(APPLE)
    set_target_properties(duckdb PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libduckdb.dylib)
else()
    set_target_properties(duckdb PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libduckdb.so)
endif()


add_dependencies(duckdb duckdb_ep)

add_library(duckdb_library SHARED imputation/triple/sum/sum.cpp imputation/include/triple/sum/sum.h imputation/triple/mul.cpp imputation/include/triple/mul.h imputation/triple/SQL_lift.cpp imputation/include/triple/SQL_lift.h imputation/include/triple/From_duckdb.h imputation/triple/From_duckdb.cpp imputation/triple/lift.cpp imputation/include/triple/lift.h imputation/triple/sum/sum_no_lift.cpp imputation/include/triple/sum/sum_no_lift.h imputation/triple/sub.cpp imputation/include/triple/sub.h imputation/ML/regression.cpp imputation/include/ML/regression.h imputation/ML/regression_predict.cpp imputation/include/ML/regression_predict.h imputation/triple/helper.cpp imputation/include/helper.h experiments/partition.cpp include/partition.h tests/test_lift.cpp tests/test_sum.cpp tests/test_mul.cpp imputation/ML/lda.cpp imputation/include/ML/lda.h imputation/ML/utils.cpp imputation/include/ML/utils.h imputation/ML/lda_impute.cpp imputation/include/ML/lda_impute.h tests/test_lda.cpp experiments/flight_partition.cpp include/flight_partition.h experiments/flight_baseline.cpp include/flight_baseline.h experiments/train_flight.cpp include/train_flight.h experiments/train_retailer.cpp include/train_retailer.h experiments/column_scalability.cpp include/column_scalability.h experiments/factorized_imputation_flight.cpp include/factorized_imputation_flight.h experiments/factorized_imputation_retailer.cpp include/factorized_imputation_retailer.h imputation/triple/sum/sum_state.cpp imputation/include/triple/sum/sum_state.h tests/test_ML.cpp imputation/triple/lift_nb.cpp imputation/triple/sum/sum_nb.cpp imputation/include/triple/sum/sum_nb_state.h imputation/triple/sum/sum_nb_no_lift.cpp imputation/include/triple/sum/sum_nb.h imputation/include/triple/sum/sum_nb_no_lift.h imputation/triple/sum/sum_nb_state.cpp experiments/flight_partition_2.cpp include/flight_partition_2.h)
target_link_libraries(duckdb_library duckdb)
target_link_libraries(duckdb_library ${lapackblas_libraries})
target_compile_options(duckdb_library PRIVATE ${LAPACK_LINKER_FLAGS})

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED)

target_include_directories(duckdb_library PRIVATE imputation/include)
target_include_directories(duckdb_library PRIVATE include)#todo merge

if(Boost_FOUND)
    target_include_directories(duckdb_library PRIVATE ${Boost_INCLUDE_DIRS})
endif()

add_dependencies(duckdb_library duckdb_ep)

add_executable(DuckDB_imputation main.cpp)
add_dependencies(DuckDB_imputation duckdb_library)
add_dependencies(DuckDB_imputation duckdb)


target_link_libraries(DuckDB_imputation PRIVATE duckdb)
target_link_libraries(DuckDB_imputation PRIVATE duckdb_library)
target_include_directories(DuckDB_imputation PRIVATE include)
target_include_directories(DuckDB_imputation PRIVATE imputation/include)

