#include <utils.h>

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