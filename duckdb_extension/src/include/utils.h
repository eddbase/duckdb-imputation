//
// Created by Massimo Perini on 10/01/2024.
//

#ifndef DUCKDB_UTILS_H
#define DUCKDB_UTILS_H

#include <duckdb.hpp>

using namespace duckdb;
namespace duckdb {


struct StructStats {
  DUCKDB_API static void Construct(BaseStatistics &stats);
  DUCKDB_API static BaseStatistics CreateUnknown(LogicalType type);
  DUCKDB_API static BaseStatistics CreateEmpty(LogicalType type);

  DUCKDB_API static const BaseStatistics *
  GetChildStats(const BaseStatistics &stats);
  DUCKDB_API static const BaseStatistics &
  GetChildStats(const BaseStatistics &stats, idx_t i);
  DUCKDB_API static BaseStatistics &GetChildStats(BaseStatistics &stats,
                                                  idx_t i);
  DUCKDB_API static void SetChildStats(BaseStatistics &stats, idx_t i,
                                       const BaseStatistics &new_stats);
  DUCKDB_API static void SetChildStats(BaseStatistics &stats, idx_t i,
                                       unique_ptr<BaseStatistics> new_stats);

  // DUCKDB_API static void Serialize(const BaseStatistics &stats, FieldWriter &writer); DUCKDB_API static BaseStatistics Deserialize(FieldReader &reader, LogicalType type);

  DUCKDB_API static string ToString(const BaseStatistics &stats);

  DUCKDB_API static void Merge(BaseStatistics &stats,
                               const BaseStatistics &other);
  DUCKDB_API static void Copy(BaseStatistics &stats,
                              const BaseStatistics &other);
  DUCKDB_API static void Verify(const BaseStatistics &stats, Vector &vector,
                                const SelectionVector &sel, idx_t count);
};

void RecursiveFlatten(Vector &vector, idx_t &count);
}


#endif // DUCKDB_UTILS_H
