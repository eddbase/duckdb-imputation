

#include <triple/From_duckdb.h>

void duckdb::RecursiveFlatten(Vector &vector, idx_t &count) {
    if (vector.GetVectorType() != VectorType::FLAT_VECTOR) {
        vector.Flatten(count);
    }

    auto internal_type = vector.GetType().InternalType();
    if (internal_type == PhysicalType::LIST) {
        auto &child_vector = ListVector::GetEntry(vector);
        auto child_vector_count = ListVector::GetListSize(vector);
        RecursiveFlatten(child_vector, child_vector_count);
    } else if (internal_type == PhysicalType::STRUCT) {
        auto &children = StructVector::GetEntries(vector);
        for (auto &child : children) {
            RecursiveFlatten(*child, count);
        }
    }
}

BaseStatistics StructStats::CreateUnknown(LogicalType type) {
    auto &child_types = StructType::GetChildTypes(type);
    BaseStatistics result(std::move(type));
    result.InitializeUnknown();
    for (idx_t i = 0; i < child_types.size(); i++) {
        result.child_stats[i].Copy(BaseStatistics::CreateUnknown(child_types[i].second));
    }
    return result;
}

void StructStats::Copy(BaseStatistics &stats, const BaseStatistics &other) {
    auto count = StructType::GetChildCount(stats.GetType());
    for (idx_t i = 0; i < count; i++) {
        stats.child_stats[i].Copy(other.child_stats[i]);
    }
}


const BaseStatistics *StructStats::GetChildStats(const BaseStatistics &stats) {
    if (stats.GetStatsType() != StatisticsType::STRUCT_STATS) {
        throw InternalException("Calling StructStats::GetChildStats on stats that is not a struct");
    }
    return stats.child_stats.get();
}

const BaseStatistics &StructStats::GetChildStats(const BaseStatistics &stats, idx_t i) {
            D_ASSERT(stats.GetStatsType() == StatisticsType::STRUCT_STATS);
    if (i >= StructType::GetChildCount(stats.GetType())) {
        throw InternalException("Calling StructStats::GetChildStats but there are no stats for this index");
    }
    return stats.child_stats[i];
}

BaseStatistics &StructStats::GetChildStats(BaseStatistics &stats, idx_t i) {
            D_ASSERT(stats.GetStatsType() == StatisticsType::STRUCT_STATS);
    if (i >= StructType::GetChildCount(stats.GetType())) {
        throw InternalException("Calling StructStats::GetChildStats but there are no stats for this index");
    }
    return stats.child_stats[i];
}
