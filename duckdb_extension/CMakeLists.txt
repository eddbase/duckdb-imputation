cmake_minimum_required(VERSION 2.8.12)

# Set extension name here
set(TARGET_NAME duckdb_imputation)

set(CMAKE_BUILD_PARALLEL_LEVEL 8)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DuckDB's extension distribution supports vcpkg. As such, dependencies can be added in ./vcpkg.json and then
# used in cmake with find_package. Feel free to remove or replace with other dependencies.
# Note that it should also be removed from vcpkg.json to prevent needlessly installing it..
find_package(OpenSSL REQUIRED)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/duckdb_imputation_extension.cpp src/triple/sum/sum.cpp src/triple/sum/sum_state.cpp src/triple/lift.cpp src/triple/sum/sum_no_lift.cpp src/triple/mul.cpp src/ML/regression.cpp src/ML/lda.cpp src/ML/qda.cpp src/ML/naive_bayes.cpp src/ML/utils.cpp src/utils.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

find_package(BLAS)
find_package(LAPACK)
if(LAPACK_FOUND AND BLAS_FOUND)
    set(lapackblas_libraries ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    panic()
endif()
target_link_libraries(${EXTENSION_NAME} ${lapackblas_libraries})
target_compile_options(${EXTENSION_NAME} PRIVATE ${LAPACK_LINKER_FLAGS})

target_link_libraries(${LOADABLE_EXTENSION_NAME} ${lapackblas_libraries})
target_compile_options(${LOADABLE_EXTENSION_NAME} PRIVATE ${LAPACK_LINKER_FLAGS})



# Link OpenSSL in both the static library as the loadable extension
target_link_libraries(${EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${LOADABLE_EXTENSION_NAME} OpenSSL::SSL OpenSSL::Crypto)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
