
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

#include <cassert>
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#ifdef HPCG_USE_NEON
#include <arm_neon.h>
#endif
#ifdef HPCG_USE_SVE
#include <arm_sve.h>
#endif
#ifdef HPCG_USE_ARMPL_SPMV
#include "armpl_sparse.h"
#endif
/*#ifndef HPCG_USE_ARMPL_SPMV
#include <armpl_sparse.h>
#endif*/


/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

   assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  const double * __restrict__ xv_r = (const double *)__builtin_assume_aligned(xv, 64);
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;
#ifndef HPCG_NO_OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (local_int_t i=0; i< nrow; ++i) {
    const double * const cur_vals = A.matrixValues[i];
    const double * __restrict__ cur_vals_r = (const double *)__builtin_assume_aligned(cur_vals, 64);
    const local_int_t * const cur_inds = A.mtxIndL[i];
    const local_int_t * __restrict__ cur_inds_r = (const local_int_t *)__builtin_assume_aligned(cur_inds, 64);
    const int cur_nnz = A.nonzerosInRow[i];

    double sum = 0.0;
    const int BS = 32; /* block size, 可调 */
    int j = 0;
    const int limit = cur_nnz - (cur_nnz % BS);

    for (; j < limit; j += BS) {
      const int base = j;
      #pragma omp simd reduction(+:sum)
      for (int t = 0; t < BS; ++t) {
        const local_int_t idx = cur_inds_r[base + t];
        sum += cur_vals_r[base + t] * xv_r[idx];
      }
    }

    /* tail */
    for (; j < cur_nnz; ++j) {
      sum += cur_vals_r[j] * xv_r[cur_inds_r[j]];
    }

    yv[i] = sum;
  }
  return 0;
  // This line and the next two lines should be removed and your version of ComputeSPMV should be used.
  //A.isSpmvOptimized = false;
  //return ComputeSPMV_ref(A, x, y);
}
