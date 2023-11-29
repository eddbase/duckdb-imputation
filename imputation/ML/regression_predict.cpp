
#include <ML/regression_predict.h>
#include <duckdb/function/scalar/nested_functions.hpp>


#include <iostream>
#include <algorithm>
#include <triple/From_duckdb.h>

//DataChunk is a set of vectors

class duckdb::BoundFunctionExpression : public duckdb::Expression {
public:
    static constexpr const ExpressionClass TYPE = ExpressionClass::BOUND_FUNCTION;

public:
    BoundFunctionExpression(LogicalType return_type, ScalarFunction bound_function,
                            vector<unique_ptr<Expression>> arguments, unique_ptr<FunctionData> bind_info,
                            bool is_operator = false);

    //! The bound function expression
    ScalarFunction function;
    //! List of child-expressions of the function
    vector<unique_ptr<Expression>> children;
    //! The bound function data (if any)
    unique_ptr<FunctionData> bind_info;
    //! Whether or not the function is an operator, only used for rendering
    bool is_operator;

public:
    bool HasSideEffects() const override;
    bool IsFoldable() const override;
    string ToString() const override;
    bool PropagatesNullValues() const override;
    hash_t Hash() const override;
    bool Equals(const BaseExpression *other) const;

    unique_ptr<Expression> Copy() override;
    void Verify() const override;

    //void Serialize(FieldWriter &writer) const override;
    //static unique_ptr<Expression> Deserialize(ExpressionDeserializationState &state, FieldReader &reader);
};

namespace duckdb {
    struct ListStats {
        DUCKDB_API static void Construct(BaseStatistics &stats);

        DUCKDB_API static BaseStatistics CreateUnknown(LogicalType type);

        DUCKDB_API static BaseStatistics CreateEmpty(LogicalType type);

        DUCKDB_API static const BaseStatistics &GetChildStats(const BaseStatistics &stats);

        DUCKDB_API static BaseStatistics &GetChildStats(BaseStatistics &stats);

        DUCKDB_API static void SetChildStats(BaseStatistics &stats, unique_ptr<BaseStatistics> new_stats);

        //DUCKDB_API static void Serialize(const BaseStatistics &stats, FieldWriter &writer);

        //DUCKDB_API static BaseStatistics Deserialize(FieldReader &reader, LogicalType type);

        DUCKDB_API static string ToString(const BaseStatistics &stats);

        DUCKDB_API static void Merge(BaseStatistics &stats, const BaseStatistics &other);

        DUCKDB_API static void Copy(BaseStatistics &stats, const BaseStatistics &other);

        DUCKDB_API static void
        Verify(const BaseStatistics &stats, Vector &vector, const SelectionVector &sel, idx_t count);
    };
}

namespace Triple {
    //actual implementation of this function
    //make sure first argument is boolean mask and second one value to update

    void ImputeHackFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state, duckdb::Vector &result) {
        //std::cout << "StructPackFunction start " << std::endl;
        //col, list of params

        idx_t rows = args.size();//n. of rows to return
        vector<Vector> &in_data = args.data;
        idx_t columns = in_data.size();
        //std::cout<<"Intermediate chunk \n";

        for(size_t i=0; i<columns;i++){
            RecursiveFlatten(args.data[i], rows);
            D_ASSERT((args.data[i]).GetVectorType() == duckdb::VectorType::FLAT_VECTOR);
        }

        auto mask = (bool *) duckdb::FlatVector::GetData(args.data[0]);//first child (N)
        auto update_col = (float *) duckdb::FlatVector::GetData(args.data[1]);//first child (N)
        auto result_ptr = (float *) result.GetData();//useless

        for(size_t i=0; i<rows; i++){
            //float res = 0;
            //state.intermediate_chunk.data[1].GetData()[i] = 666;
            if(mask[i]){
                float new_val = 12345;
                state.intermediate_chunk.SetValue(1, i, Value(12345));
                //for(size_t j=2; j<columns;j++){
                //    new_val += duckdb::FlatVector::GetData(args.data[j])[i];
                //}
                //std::cout<<"CUrrent value: "<<update_col[i]<<"\n";
                update_col[i] = new_val;
                result_ptr[i] = new_val;
                //std::cout<<"New value: "<<update_col[i];
                //state.intermediate_chunk.SetValue(1, i, Value((float)666.4));
            }
            else
                result_ptr[i] = update_col[i];
        }
    }

    //Returns the datatype used by this function
    duckdb::unique_ptr<duckdb::FunctionData>
    ImputeHackBind(duckdb::ClientContext &context, duckdb::ScalarFunction &function,
                 duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> &arguments) {
        function.return_type = LogicalType::FLOAT;
        return duckdb::make_uniq<duckdb::VariableReturnBindData>(function.return_type);
    }

    //Generate statistics for this function. Given input type ststistics (mainly min and max for every attribute), returns the output statistics
    duckdb::unique_ptr<duckdb::BaseStatistics>
    ImputeHackStats(duckdb::ClientContext &context, duckdb::FunctionStatisticsInput &input) {
        auto &child_stats = input.child_stats;
        auto &expr = input.expr;
        auto stats = duckdb::BaseStatistics::CreateUnknown(expr.return_type);
        return stats.ToUnique();
    }
}
