#define DUCKDB_EXTENSION_MAIN

#include "duckdb_imputation_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>


//
#include <triple/sum/sum.h>
#include <triple/sum/sum_state.h>
#include <triple/lift.h>
#include <triple/mul.h>
#include <triple/sum/sum_no_lift.h>
#include <ML/lda.h>

#include <duckdb/function/scalar/nested_functions.hpp>


namespace duckdb {

inline void DuckdbImputationScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "DuckdbImputation "+name.GetString()+" 🐥");;
        });
}

inline void DuckdbImputationOpenSSLVersionScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
    auto &name_vector = args.data[0];
    UnaryExecutor::Execute<string_t, string_t>(
	    name_vector, result, args.size(),
	    [&](string_t name) {
			return StringVector::AddString(result, "DuckdbImputation " + name.GetString() +
                                                     ", my linked OpenSSL version is " +
                                                     OPENSSL_VERSION_TEXT );;
        });
}

static void LoadInternal(DatabaseInstance &instance) {
    // Register a scalar function
    //here register aggregate and scalar functions
    auto duckdb_imputation_scalar_function = ScalarFunction("duckdb_imputation", {LogicalType::VARCHAR}, LogicalType::VARCHAR, DuckdbImputationScalarFun);
    ExtensionUtil::RegisterFunction(instance, duckdb_imputation_scalar_function);

    // Register another scalar function
    auto duckdb_imputation_openssl_version_scalar_function = ScalarFunction("duckdb_imputation_openssl_version", {LogicalType::VARCHAR},
                                                LogicalType::VARCHAR, DuckdbImputationOpenSSLVersionScalarFun);
    ExtensionUtil::RegisterFunction(instance, duckdb_imputation_openssl_version_scalar_function);

    //add my funcs


    auto sum_triple_func = duckdb::AggregateFunction("sum_triple", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, duckdb::AggregateFunction::StateSize<Triple::SumState>,
                                              duckdb::AggregateFunction::StateInitialize<Triple::SumState, Triple::StateFunction>, Triple::Sum,
                                              Triple::SumStateCombine, Triple::SumStateFinalize, nullptr, Triple::SumBind,
                                              duckdb::AggregateFunction::StateDestroy<Triple::SumState, Triple::StateFunction>, nullptr, nullptr);

    ExtensionUtil::RegisterFunction(instance, sum_triple_func);


    duckdb::ScalarFunction to_cofactor_func("to_cofactor", {}, duckdb::LogicalTypeId::STRUCT, Triple::CustomLift, Triple::CustomLiftBind, nullptr,
                                       Triple::CustomLiftStats);
    to_cofactor_func.varargs = duckdb::LogicalType::ANY;
    to_cofactor_func.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    to_cofactor_func.serialize = duckdb::VariableReturnBindData::Serialize;
    to_cofactor_func.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, to_cofactor_func);
    //duckdb::CreateScalarFunctionInfo custom_lift_info(custom_lift);
    //info.schema = DEFAULT_SCHEMA;

    duckdb::ScalarFunction mul_fun("multiply_triple", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, Triple::MultiplyFunction, Triple::MultiplyBind, nullptr,
                               Triple::MultiplyStats);
    mul_fun.varargs = duckdb::LogicalType::ANY;
    mul_fun.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    mul_fun.serialize = duckdb::VariableReturnBindData::Serialize;
    mul_fun.deserialize = duckdb::VariableReturnBindData::Deserialize;
    //duckdb::CreateScalarFunctionInfo info(fun);
    //info.schema = DEFAULT_SCHEMA;
    ExtensionUtil::RegisterFunction(instance, mul_fun);



    int n_cont_func = 20;
    int n_cat_func = 20;

    for (int curr_cont_params=0; curr_cont_params<n_cont_func; curr_cont_params++) {
      for (int curr_cat_params = 0; curr_cat_params < n_cat_func;
           curr_cat_params++) {

        if(curr_cont_params == 0 && curr_cat_params == 0)
          continue ;

        duckdb::vector<duckdb::LogicalType> args_sum_no_lift = {};
        for (int i = 0; i < curr_cont_params; i++)
          args_sum_no_lift.push_back(duckdb::LogicalType::FLOAT);
        for (int i = 0; i < curr_cat_params; i++)
          args_sum_no_lift.push_back(duckdb::LogicalType::INTEGER);
        std::string xx = std::to_string(curr_cont_params)+"_"+std::to_string(curr_cat_params);

        auto sum_no_lift = duckdb::AggregateFunction(
            "sum_to_triple_"+xx, args_sum_no_lift, duckdb::LogicalTypeId::STRUCT,
            duckdb::AggregateFunction::StateSize<Triple::SumState>,
            duckdb::AggregateFunction::StateInitialize<Triple::SumState,
                                                       Triple::StateFunction>,
            Triple::SumNoLift, Triple::SumStateCombine,
            Triple::SumStateFinalize, nullptr, Triple::SumNoLiftBind,
            duckdb::AggregateFunction::StateDestroy<Triple::SumState,
                                                    Triple::StateFunction>,
            nullptr, nullptr);

        sum_no_lift.varargs = duckdb::LogicalType::ANY;
        sum_no_lift.null_handling =
            duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        ExtensionUtil::RegisterFunction(instance, sum_no_lift);
      }
    }

    duckdb::ScalarFunction lda_train_func = ScalarFunction("lda_train", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::LIST, lda_train, lda_train_bind, nullptr);

    lda_train_func.varargs = duckdb::LogicalType::ANY;
    lda_train_func.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    lda_train_func.serialize = duckdb::VariableReturnBindData::Serialize;
    lda_train_func.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, lda_train_func);

    //select lda_train({'N': 5, 'lin_agg': [15.0, 17.0], 'quad_agg': [59.0, 71.0, 91.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}, 1, 0, false);

    duckdb::ScalarFunction lda_predict("lda_predict", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::INTEGER, LDA_impute, LDA_impute_bind, nullptr,
                                       LDA_impute_stats);
    lda_predict.varargs = duckdb::LogicalType::ANY;
    lda_predict.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    lda_predict.serialize = duckdb::VariableReturnBindData::Serialize;
    lda_predict.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, lda_predict);

}

//select lda_train({'N': 2, 'lin_agg': [6.0], 'quad_agg': [26.0], 'lin_cat': [[{'key': 5, 'value': 1.0}, {'key': 9, 'value': 1.0}]], 'quad_num_cat': [[{'key': 5, 'value': 1.0}, {'key': 9, 'value': 5.0}]], 'quad_cat': [[{'key1': 5, 'key2': 5, 'value': 1.0}, {'key1': 9, 'key2': 9, 'value': 1.0}]]}, 1, 0, false);

void DuckdbImputationExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string DuckdbImputationExtension::Name() {
	return "duckdb_imputation";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void duckdb_imputation_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::DuckdbImputationExtension>();
}

DUCKDB_EXTENSION_API const char *duckdb_imputation_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
