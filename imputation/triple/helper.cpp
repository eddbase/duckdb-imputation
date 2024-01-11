

#include "helper.h"

#include <triple/sum/sum.h>
#include <triple/mul.h>
#include <triple/SQL_lift.h>
#include <triple/lift.h>
#include <triple/sum/sum_no_lift.h>
#include <triple/sum/sum_state.h>
#include <triple/sub.h>
#include <ML/lda_impute.h>

#include <duckdb/function/scalar/nested_functions.hpp>
#include <duckdb/function/aggregate_function.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

namespace ML_lib {

    void register_functions(duckdb::ClientContext &context, const std::vector<size_t> &n_con_columns, const std::vector<size_t> &n_cat_columns){

        auto function = duckdb::AggregateFunction("sum_triple", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, duckdb::AggregateFunction::StateSize<Triple::SumState>,
                                                  duckdb::AggregateFunction::StateInitialize<Triple::SumState, Triple::StateFunction>, Triple::Sum,
                                                  Triple::SumStateCombine, Triple::SumStateFinalize, nullptr, Triple::SumBind,
                                                  duckdb::AggregateFunction::StateDestroy<Triple::SumState, Triple::StateFunction>, nullptr, nullptr);

        duckdb::UDFWrapper::RegisterAggrFunction(function, context);
        std::set<std::pair<int, int>> fun_type_sizes;//remove duplicates
        for(size_t k=0; k<n_con_columns.size(); k++) {//for each table
            fun_type_sizes.insert(std::pair<int, int>(n_con_columns[k], n_cat_columns[k]));
        }
        int s = 1;
        for (auto const &x : fun_type_sizes){
            duckdb::vector<duckdb::LogicalType> args_sum_no_lift = {};
            for (int i = 0; i < x.first; i++)
                args_sum_no_lift.push_back(duckdb::LogicalType::FLOAT);
            for (int i = 0; i < x.second; i++)
                args_sum_no_lift.push_back(duckdb::LogicalType::INTEGER);
            std::string xx = "";
            if (fun_type_sizes.size() > 1){
                xx = std::to_string(s);
                s++;
            }
            auto sum_no_lift = duckdb::AggregateFunction("sum_to_triple"+xx, args_sum_no_lift,
                                                         duckdb::LogicalTypeId::STRUCT,
                                                         duckdb::AggregateFunction::StateSize<Triple::SumState>,
                                                         duckdb::AggregateFunction::StateInitialize<Triple::SumState, Triple::StateFunction>,
                                                         Triple::SumNoLift,
                                                         Triple::SumStateCombine, Triple::SumStateFinalize, nullptr,
                                                         Triple::SumNoLiftBind,
                                                         duckdb::AggregateFunction::StateDestroy<Triple::SumState, Triple::StateFunction>,
                                                         nullptr, nullptr);
            sum_no_lift.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
            sum_no_lift.varargs = duckdb::LogicalType::FLOAT;
            duckdb::UDFWrapper::RegisterAggrFunction(sum_no_lift, context, duckdb::LogicalType::FLOAT);
        }

        //define multiply func.

        duckdb::ScalarFunction fun("multiply_triple", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::STRUCT, Triple::MultiplyFunction, Triple::MultiplyBind, nullptr,
                                   Triple::MultiplyStats);
        fun.varargs = duckdb::LogicalType::ANY;
        fun.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        fun.serialize = duckdb::VariableReturnBindData::Serialize;
        fun.deserialize = duckdb::VariableReturnBindData::Deserialize;
        duckdb::CreateScalarFunctionInfo info(fun);
        info.schema = DEFAULT_SCHEMA;
        context.RegisterFunction(info);

        //impute LDA

        duckdb::ScalarFunction lda_predict("lda_predict", {duckdb::LogicalType::ANY}, duckdb::LogicalTypeId::INTEGER, LDA_impute, LDA_impute_bind, nullptr,
                                   LDA_impute_stats);
        lda_predict.varargs = duckdb::LogicalType::ANY;
        lda_predict.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        lda_predict.serialize = duckdb::VariableReturnBindData::Serialize;
        lda_predict.deserialize = duckdb::VariableReturnBindData::Deserialize;
        duckdb::CreateScalarFunctionInfo lda_p(lda_predict);
        lda_p.schema = DEFAULT_SCHEMA;
        context.RegisterFunction(lda_p);

        //sub triples

        duckdb::child_list_t<duckdb::LogicalType> struct_children;
        struct_children.emplace_back("N", duckdb::LogicalType::INTEGER);
        struct_children.emplace_back("lin_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));
        struct_children.emplace_back("quad_agg", duckdb::LogicalType::LIST(duckdb::LogicalType::FLOAT));

        auto struct_type = duckdb::LogicalType::STRUCT(struct_children);

        duckdb::ScalarFunction fun_sub("sub_triple", {struct_type, struct_type}, duckdb::LogicalTypeId::STRUCT, Triple::TripleSubFunction, Triple::TripleSubBind, nullptr,
                                       Triple::TripleSubStats);
        fun_sub.varargs = struct_type;
        fun_sub.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        fun_sub.serialize = duckdb::VariableReturnBindData::Serialize;
        fun_sub.deserialize = duckdb::VariableReturnBindData::Deserialize;
        duckdb::CreateScalarFunctionInfo info_sub(fun_sub);
        info_sub.schema = DEFAULT_SCHEMA;
        context.RegisterFunction(info_sub);


        //custom lift

        duckdb::ScalarFunction custom_lift("to_cofactor", {}, duckdb::LogicalTypeId::STRUCT, Triple::CustomLift, Triple::CustomLiftBind, nullptr,
                                           Triple::CustomLiftStats);
        custom_lift.varargs = duckdb::LogicalType::ANY;
        custom_lift.null_handling = duckdb::FunctionNullHandling::SPECIAL_HANDLING;
        custom_lift.serialize = duckdb::VariableReturnBindData::Serialize;
        custom_lift.deserialize = duckdb::VariableReturnBindData::Deserialize;
        duckdb::CreateScalarFunctionInfo custom_lift_info(custom_lift);
        info.schema = DEFAULT_SCHEMA;
        context.RegisterFunction(custom_lift_info);


    }

} // Triple
