// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @omp_decl_tar_host
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(host)>} {
func.func @omp_decl_tar_host() -> () attributes {omp.declare_target = #omp<device_type(host)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_nohost
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(nohost)>} {
func.func @omp_decl_tar_nohost() -> () attributes {omp.declare_target = #omp<device_type(nohost)>} {
  return
}

// CHECK-LABEL: func @omp_decl_tar_any
// CHECK-SAME: {{.*}} attributes {omp.declare_target = #omp<device_type(any)>} {
func.func @omp_decl_tar_any() -> () attributes {omp.declare_target = #omp<device_type(any)>} {
  return
}