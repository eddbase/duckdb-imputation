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

#include <triple/sum/sum_to_nb_agg.h>
#include <triple/sum/sum_nb_agg.h>
#include <triple/lift_to_nb_agg.h>
#include <triple/mul_nb.h>


#include <ML/lda.h>
#include <ML/regression.h>
#include <ML/naive_bayes.h>
#include <ML/qda.h>


#include <duckdb/function/scalar/nested_functions.hpp>


namespace duckdb {

inline void QuackScalarFun(DataChunk &args, ExpressionState &state, Vector &result) {
  auto &name_vector = args.data[0];
  UnaryExecutor::Execute<string_t, string_t>(
      name_vector, result, args.size(),
      [&](string_t name) {
        return StringVector::AddString(result, "Quack "+name.GetString()+" 🐥");;
      });
}

static void load_ring(DatabaseInstance &instance) {

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



    int n_cont_func = 20;//todo make define
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
}

static void load_nb_ring(DatabaseInstance &instance) {

    auto sum_triple_func = duckdb::AggregateFunction("sum_nb_agg", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, duckdb::AggregateFunction::StateSize<Triple::SumState>,
                                                     duckdb::AggregateFunction::StateInitialize<Triple::SumState, Triple::StateFunction>, Triple::sum_nb_agg,
                                                     Triple::SumStateCombine, Triple::SumStateFinalize, nullptr, Triple::sum_nb_agg_bind,
                                                     duckdb::AggregateFunction::StateDestroy<Triple::SumState, Triple::StateFunction>, nullptr, nullptr);

    ExtensionUtil::RegisterFunction(instance, sum_triple_func);


    duckdb::ScalarFunction to_cofactor_func("to_nb_agg", {}, duckdb::LogicalTypeId::STRUCT, Triple::to_nb_lift, Triple::to_nb_lift_bind, nullptr);
    to_cofactor_func.varargs = duckdb::LogicalType::ANY;
    to_cofactor_func.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    to_cofactor_func.serialize = duckdb::VariableReturnBindData::Serialize;
    to_cofactor_func.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, to_cofactor_func);
    //duckdb::CreateScalarFunctionInfo custom_lift_info(custom_lift);
    //info.schema = DEFAULT_SCHEMA;

    duckdb::ScalarFunction mul_fun("multiply_nb_agg", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, Triple::multiply_nb, Triple::multiply_nb_bind, nullptr);
    mul_fun.varargs = duckdb::LogicalType::ANY;
    mul_fun.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    mul_fun.serialize = duckdb::VariableReturnBindData::Serialize;
    mul_fun.deserialize = duckdb::VariableReturnBindData::Deserialize;
    //duckdb::CreateScalarFunctionInfo info(fun);
    //info.schema = DEFAULT_SCHEMA;
    ExtensionUtil::RegisterFunction(instance, mul_fun);



    int n_cont_func = 20;//todo make define
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
            "sum_to_nb_agg_"+xx, args_sum_no_lift, duckdb::LogicalTypeId::STRUCT,
            duckdb::AggregateFunction::StateSize<Triple::SumState>,
            duckdb::AggregateFunction::StateInitialize<Triple::SumState,
                                                       Triple::StateFunction>,
            Triple::sum_to_nb_agg, Triple::SumStateCombine,
            Triple::SumStateFinalize, nullptr, Triple::sum_to_nb_agg_bind,
            duckdb::AggregateFunction::StateDestroy<Triple::SumState,
                                                    Triple::StateFunction>,
            nullptr, nullptr);

        sum_no_lift.varargs = duckdb::LogicalType::ANY;
        sum_no_lift.null_handling =
            duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        ExtensionUtil::RegisterFunction(instance, sum_no_lift);
      }
    }
}

static void load_ml(DatabaseInstance &instance){

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


    duckdb::ScalarFunction linreg_train_func = ScalarFunction("linreg_train", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::LIST, ML::ridge_linear_regression, ML::ridge_linear_regression_bind, nullptr);
    linreg_train_func.varargs = duckdb::LogicalType::ANY;
    linreg_train_func.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    linreg_train_func.serialize = duckdb::VariableReturnBindData::Serialize;
    linreg_train_func.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, linreg_train_func);

    //select lda_train({'N': 5, 'lin_agg': [15.0, 17.0], 'quad_agg': [59.0, 71.0, 91.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}, 1, 0, false);

    duckdb::ScalarFunction linreg_predict("linreg_predict", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::INTEGER, ML::linreg_impute, ML::linreg_impute_bind, nullptr, nullptr, ML::regression_init_state);
    linreg_predict.varargs = duckdb::LogicalType::ANY;
    linreg_predict.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    linreg_predict.serialize = duckdb::VariableReturnBindData::Serialize;
    linreg_predict.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, linreg_predict);


    duckdb::ScalarFunction qda_train_func = ScalarFunction("qda_train", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::LIST, ML::qda_train, ML::qda_train_bind, nullptr);
    qda_train_func.varargs = duckdb::LogicalType::ANY;
    qda_train_func.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    qda_train_func.serialize = duckdb::VariableReturnBindData::Serialize;
    qda_train_func.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, qda_train_func);

    //select lda_train({'N': 5, 'lin_agg': [15.0, 17.0], 'quad_agg': [59.0, 71.0, 91.0], 'lin_cat': [], 'quad_num_cat': [], 'quad_cat': []}, 1, 0, false);

    duckdb::ScalarFunction qda_predict("qda_predict", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::INTEGER, ML::qda_impute, ML::qda_impute_bind, nullptr);
    qda_predict.varargs = duckdb::LogicalType::ANY;
    qda_predict.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    qda_predict.serialize = duckdb::VariableReturnBindData::Serialize;
    qda_predict.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, qda_predict);


    duckdb::ScalarFunction nb_train("nb_train", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::LIST, ML::nb_train, ML::nb_train_bind, nullptr);
    nb_train.varargs = duckdb::LogicalType::ANY;
    nb_train.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    nb_train.serialize = duckdb::VariableReturnBindData::Serialize;
    nb_train.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, nb_train);

    duckdb::ScalarFunction nb_predict("nb_predict", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::INTEGER, ML::nb_impute, ML::nb_impute_bind, nullptr);
    nb_predict.varargs = duckdb::LogicalType::ANY;
    nb_predict.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
    nb_predict.serialize = duckdb::VariableReturnBindData::Serialize;
    nb_predict.deserialize = duckdb::VariableReturnBindData::Deserialize;
    ExtensionUtil::RegisterFunction(instance, nb_predict);
}


//select lda_train({'N': 2, 'lin_agg': [6.0], 'quad_agg': [26.0], 'lin_cat': [[{'key': 5, 'value': 1.0}, {'key': 9, 'value': 1.0}]], 'quad_num_cat': [[{'key': 5, 'value': 1.0}, {'key': 9, 'value': 5.0}]], 'quad_cat': [[{'key1': 5, 'key2': 5, 'value': 1.0}, {'key1': 9, 'key2': 9, 'value': 1.0}]]}, 1, 0, false);

void DuckdbImputationExtension::Load(DuckDB &db) {
    load_ring(*db.instance);
    load_nb_ring(*db.instance);
    load_ml(*db.instance);

    auto quack_scalar_function = ScalarFunction("quack", {LogicalType::VARCHAR}, LogicalType::VARCHAR, QuackScalarFun);
    ExtensionUtil::RegisterFunction(*db.instance, quack_scalar_function);

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
