cmake_minimum_required(VERSION 3.1)
project(DuckDB_imputation)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

set(ENABLE_DOCTESTS ON)

ExternalProject_Add(
        duckdb_ep
        PREFIX "${CMAKE_BINARY_DIR}/duckdb_extension/duckdb"
        SOURCE_DIR "${CMAKE_SOURCE_DIR}/duckdb_extension/duckdb"
        INSTALL_DIR "${CMAKE_BINARY_DIR}/duckdb_extension/duckdb/install"
        CMAKE_ARGS
        -G${CMAKE_GENERATOR}
        -DCMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}
        -DCMAKE_BUILD_TYPE=RelWithDebInfo
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CXX_FLAGS=-std=c++17
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/duckdb_extension/duckdb/install
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

add_library(duckdb SHARED IMPORTED)
set_property(TARGET duckdb PROPERTY IMPORTED_LOCATION ${DUCKDB_LIBRARY_PATH})
set_property(TARGET duckdb APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${DUCKDB_INCLUDE_DIR})

if(APPLE)
    set_target_properties(duckdb PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libduckdb.dylib)
else()
    set_target_properties(duckdb PROPERTIES IMPORTED_LOCATION ${INSTALL_DIR}/lib/libduckdb.so)
endif()


add_dependencies(duckdb duckdb_ep)

add_library(duckdb_library SHARED imputation/triple/sub.cpp imputation/include/sum_sub.h imputation/algorithms/partition.cpp imputation/include/partition.h imputation/algorithms/flight_partition.cpp imputation/include/flight_partition.h imputation/algorithms/flight_baseline.cpp imputation/include/flight_baseline.h imputation/triple/sum_nb.cpp imputation/algorithms/flight_partition_2.cpp imputation/include/flight_partition_2.h)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED)

target_include_directories(duckdb_library PRIVATE imputation/include)

target_link_libraries(duckdb_library PRIVATE duckdb)

add_executable(DuckDB_imputation main.cpp)
target_include_directories(DuckDB_imputation PRIVATE include)



add_dependencies(duckdb_library duckdb_ep)

add_dependencies(DuckDB_imputation duckdb_library)
add_dependencies(DuckDB_imputation duckdb)


target_link_libraries(DuckDB_imputation PRIVATE duckdb)
target_link_libraries(DuckDB_imputation PRIVATE duckdb_library)
target_include_directories(DuckDB_imputation PRIVATE include)
target_include_directories(DuckDB_imputation PRIVATE imputation/include)

