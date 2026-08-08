#ifndef LG_CORE_H_
#define LG_CORE_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/// Maximum possible Tensor rank
/// All tensors will have an array of this size to store
/// dims, so  keep this to a minimum.
#ifndef LG_MAX_RANK
#   define LG_MAX_RANK 8
#endif // LG_MAX_RANK

/// The number of tensors tracked by `lg_nditer`.
#ifndef LG_N_TRACKED_TENSORS
#   define LG_N_TRACKED_TENSORS 4
#endif // LG_N_TRACKED_TENSORS 4

/// Type to back Tensor data
#ifndef lg_scalar
#   define lg_scalar float
#endif // lg_scalar

#if defined(__has_feature) && __has_feature(nullability)
#   define lg_nullable _Nullable
#else
#   define lg_nullable 
#endif // defined(__has_attribute) && __has_attribute(nullability)

#define lg_nil(T) (T){0}

/// Bounds checking
#ifdef __cplusplus
#   define lg_check_bounds(x) /* nothing */
#   define lg_check_bounds_nullable(x) /* nothing */
#else
#   if defined(__clang__) && __has_attribute(counted_by)
#       define lg_check_bounds(x) __attribute__((counted_by(x)))
#       define lg_check_bounds_nullable(x) __attribute__((counted_by_or_null(x)))
#   elif defined(__GNUC__) && (__GNUC__ >= 16) // Pointer support introduced in GCC 16
#       define lg_check_bounds(x) __attribute__((counted_by(x)))
#       define lg_check_bounds_nullable(x) __attribute__((counted_by_or_null(x)))
#   else
#       define lg_check_bounds(x) /* nothing */
#       define lg_check_bounds_nullable(x) /* nothing */
#   endif
#endif

#if defined(__has_attribute) && __has_attribute(always_inline)
#   define lg_force_inline __attribute__((always_inline)) inline
#else
#   define lg_force_inline inline
#endif // defined(__has_attribute) && __has_attribute(always_inline)
    

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// Define status codes with a string reflection table
///
////////////////////////////////////////////////////////////////////////////////
 
#define lg_define_status_kinds \
    LG_X(OK), \
    LG_X(InvalidArgument), \
    LG_X(InvalidRank), \
    LG_X(ShapeMismatch), \
    LG_X(StrideMismatch), \
    LG_X(Overflow), \
    LG_X(NotFound), \
    LG_X(Duplicate), \
    LG_X(UnsupportedOpcode), \
    LG_X(OutOfMemory), \
    LG_X(OutOfBounds), \
    LG_X(UnexpectedNaN),

#define LG_X(x) LG_StatusKind_##x
typedef enum
LG_StatusKind {
    lg_define_status_kinds
} LG_StatusKind;
#undef LG_X

// TODO: maybe these should be lg_str8s
#define LG_X(x) [LG_StatusKind_##x] = (const uint8_t*)#x
static const uint8_t *LG_STATUS_KIND_CSTRING_TABLE[] = {
    lg_define_status_kinds
};
#undef LG_X

#define lg_status_kind_as_cstring(status) LG_STATUS_KIND_CSTRING_TABLE[(status)]


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// Core program data structures
///
////////////////////////////////////////////////////////////////////////////////

typedef enum
LG_LayoutKind {
    LG_LayoutKind_RowMajor,
    LG_LayoutKind_ColumnMajor,
} LG_LayoutKind;

/// Tensor shape descriptor
typedef struct
LG_StridedDesc {
    /// The rank of the tensor.
    /// Must be less than LG_MAX_RANK.
    size_t rank;

    /// The dimensionality of the tensor.
    size_t dim[LG_MAX_RANK];

    /// The strides of the tensor.
    /// The order of this array must match that of `dim`.
    size_t strides[LG_MAX_RANK];
} LG_StridedDesc;

/// Tracks the coordinates of LG_N_TRACKED_TENSORS tensors.
///
/// All tensors in a single iter must be both broadcasted and
/// have their dims sorted in descending order.
typedef struct
LG_NDIter {
    size_t          coords[LG_MAX_RANK];
    LG_StridedDesc  descs[LG_N_TRACKED_TENSORS];
    size_t          indices[LG_N_TRACKED_TENSORS];
    size_t          n_tracked_dims;
} LG_NDIter;

#define lg_mkshape_count_args(...) (sizeof((uint8_t[]){__VA_ARGS__}))
#define lg_mkshape(...) (LG_StridedDesc){ .rank = lg_mkshape_count_args(__VA_ARGS__), .dim = {__VA_ARGS__} }

/// Increment the coordinate `axis` on `iter` and update offsets.
/// 
/// Does not perform any bounds checking.
bool 
lg_nditer_increment(LG_NDIter *iter, size_t axis);

/// Recomputes the indices in `iter` according to its `coords`.
///
/// If you want to "jump" to a specific coordinate in a tensor, this is the
/// easiest way to do it.
void 
lg_nditer_goto(LG_NDIter *iter, size_t *coords);

/// Infers the broadcasted logical dimensions between `descs`.
LG_StatusKind 
lg_infer_broadcasted_dims(
    size_t *lg_nullable out_rank,
    size_t *lg_nullable out_dim,
    const LG_StridedDesc **descs,
    size_t n_descs
);

/// Create a shared virtual contraction space between `descs` such that element wise accumulators
/// may function according to broadcast semantics.
LG_StatusKind 
lg_create_broadcast_space(LG_StridedDesc **descs, size_t n_descs);

/// Computes the dimensions of a contraction between `x0` and `x1`
///
/// Does not compute strides.
LG_StatusKind 
lg_infer_contracted_dims(
    size_t *lg_nullable out_rank,
    size_t *lg_nullable out_dim,
    const LG_StridedDesc *x0,
    const LG_StridedDesc *x1,
    size_t n_contracted_axes,
    size_t n_batch_axes
);

/// Contracts the dimensions of `y`, inferring the contracted dimensions.
///
/// The contracted dimensions must be aligned at the beginning of `x0`, and `x1` with batch dimensions
/// following.
LG_StatusKind 
lg_create_contraction_space(LG_StridedDesc *y, LG_StridedDesc *x0, LG_StridedDesc *x1, size_t n_batch_axes);

/// Sort axes such that the primary is unit stride first.
///
/// Inputs to this function MUST be broadcasted.
///
/// The first tensor in the set is considered the 
/// "primary."
/// The primary tensor is the one that dictates the optimized plan
/// plan for the other tensors. In that, this is the tensor where it is 
/// guaranteed that the contiguous dimension (the dimension with the unit stride)
/// will be accessed sequentially in memory.
LG_StatusKind 
lg_sort_axes(LG_StridedDesc **descs, size_t n_descs);

/// Coalesce tensor axes to be as flat as possible.
///
/// Inputs to this function MUST be broadcasted AND sorted from least to greatest
/// using `LG_SortAxes`.
LG_StatusKind 
lg_coalesce_axes(LG_StridedDesc **descs, size_t n_descs);

/// Compute the size in bytes of a tensor's data buffer.
size_t 
lg_desc_size_in_bytes(LG_StridedDesc desc);

/// Copy a vector value to the dim `copy_to_dim`.
void 
lg_copy_vector_to_axis(LG_StridedDesc desc, lg_scalar *restrict dest, const lg_scalar *vector, size_t copy_to_axis);

/// Lays out a tensor with pre-populated `dim` and `rank` with the strides to be stored in
/// the order in `layout`. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `unit_align` if `unit_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
LG_StatusKind 
lg_desc_compute_strides(LG_StridedDesc *desc, LG_LayoutKind layout, size_t unit_align);

/// Returns true if a tensor is isotropic.
/// 
/// Tensors with a rank of zero, and all scalars are considered isotropic,
/// while all vectors are considered anisotropic.
bool 
lg_desc_is_isotropic(LG_StridedDesc desc);

#endif // LG_CORE_H_
