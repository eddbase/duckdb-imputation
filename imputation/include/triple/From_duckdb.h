

#ifndef DUCKDB_FROM_DUCKDB_H
#define DUCKDB_FROM_DUCKDB_H

#include <duckdb.hpp>
using namespace duckdb;
namespace duckdb {


    struct StructStats {
        DUCKDB_API static void Construct(BaseStatistics &stats);
        DUCKDB_API static BaseStatistics CreateUnknown(LogicalType type);
        DUCKDB_API static BaseStatistics CreateEmpty(LogicalType type);

        DUCKDB_API static const BaseStatistics *GetChildStats(const BaseStatistics &stats);
        DUCKDB_API static const BaseStatistics &GetChildStats(const BaseStatistics &stats, idx_t i);
        DUCKDB_API static BaseStatistics &GetChildStats(BaseStatistics &stats, idx_t i);
        DUCKDB_API static void SetChildStats(BaseStatistics &stats, idx_t i, const BaseStatistics &new_stats);
        DUCKDB_API static void SetChildStats(BaseStatistics &stats, idx_t i, unique_ptr<BaseStatistics> new_stats);

        //DUCKDB_API static void Serialize(const BaseStatistics &stats, FieldWriter &writer);
        //DUCKDB_API static BaseStatistics Deserialize(FieldReader &reader, LogicalType type);

        DUCKDB_API static string ToString(const BaseStatistics &stats);

        DUCKDB_API static void Merge(BaseStatistics &stats, const BaseStatistics &other);
        DUCKDB_API static void Copy(BaseStatistics &stats, const BaseStatistics &other);
        DUCKDB_API static void Verify(const BaseStatistics &stats, Vector &vector, const SelectionVector &sel, idx_t count);
    };

/*
    struct VariableReturnBindData : public FunctionData {
        LogicalType stype;

        explicit VariableReturnBindData(LogicalType stype_p) : stype(std::move(stype_p)) {
        }

        unique_ptr<FunctionData> Copy() const override {
            return make_uniq<VariableReturnBindData>(stype);
        }

        bool Equals(const FunctionData &other_p) const override {
            auto &other = (const VariableReturnBindData &) other_p;
            return stype == other.stype;
        }

        static void Serialize(FieldWriter &writer, const FunctionData *bind_data_p, const ScalarFunction &function) {
                    D_ASSERT(bind_data_p);
            auto &info = bind_data_p->Cast<VariableReturnBindData>();
            writer.WriteSerializable(info.stype);
        }

        static unique_ptr<FunctionData> Deserialize(ClientContext &context, FieldReader &reader,
                                                    ScalarFunction &bound_function) {
            auto stype = reader.ReadRequiredSerializable<LogicalType, LogicalType>();
            return make_uniq<VariableReturnBindData>(std::move(stype));
        }
    };
*/

    void RecursiveFlatten(Vector &vector, idx_t &count);

}
#endif //DUCKDB_FROM_DUCKDB_H
