#ifndef LG_AFFINE_H_
#define LG_AFFINE_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>
#include <stdint.h>

/// Hyperplane representation of a convex polyhedron bounded by
/// the inequalities Ax <= b; stored row-major
typedef struct
LG_HPolyhedron {
    uint8_t  n_rows, n_cols;
    int64_t *data;
    
    // what this would look like with separate pointers:
    // int64_t  *A; // R^(rows, cols)
    // int64_t  *b; // R^(rows)
} LG_HPolyhedron;

/// Affine map representing the equation y = Ax + b
/// stored row-major
typedef struct
LG_AffineTransform {
    // TODO: dynamically allocate these
    uint8_t  n_rows, n_cols;
    int64_t  A[LG_MAX_RANK * LG_MAX_RANK]; // R^(rows, cols)
    int64_t  b[LG_MAX_RANK]; // R^(rows)
} LG_AffineTransform;

#define lg_hpoly_get_A(hpoly) ((hpoly)->data)
#define lg_hpoly_get_b(hpoly) ((hpoly)->data + ((size_t)(hpoly)->n_rows * (hpoly)->n_cols))

LG_StatusKind
lg_hpoly_make_parallelotope(LG_Arena *arena, LG_HPolyhedron *out_hpoly, uint8_t rank, int64_t *lower, int64_t *upper);

LG_AffineTransform
lg_atran_strided_from_shape(const LG_LogicalShape *shape, LG_LayoutKind layout, uint32_t unit_align);

void
lg_atran_apply(const LG_AffineTransform *tran, const int64_t x[static LG_MAX_RANK], int64_t y[static LG_MAX_RANK]);

#endif //LG_AFFINE_H_c
