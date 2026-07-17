#ifndef LG_CORE_H_
#define LG_CORE_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <libgrad/internal/fnv.h>

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

/// Bounds checking
#ifdef __cplusplus
#   define LG_CHECK_BOUNDS(x) /* nothing */
#   define LG_CHECK_BOUNDS_NULLABLE(x) /* nothing */
#else
#   if defined(__clang__) && __has_attribute(counted_by)
#       define LG_CHECK_BOUNDS(x) __attribute__((counted_by(x)))
#       define LG_CHECK_BOUNDS_NULLABLE(x) __attribute__((counted_by_or_null(x)))
#   elif defined(__GNUC__) && (__GNUC__ >= 16) // Pointer support introduced in GCC 16
#       define LG_CHECK_BOUNDS(x) __attribute__((counted_by(x)))
#       define LG_CHECK_BOUNDS_NULLABLE(x) __attribute__((counted_by_or_null(x)))
#   else
#       define LG_CHECK_BOUNDS(x) /* nothing */
#       define LG_CHECK_BOUNDS_NULLABLE(x) /* nothing */
#   endif
#endif

enum lg_status {
    LG_STATUS_OK = 0,
    LG_STATUS_INVALID_RANK,
    LG_STATUS_SHAPE_MISMATCH,
    LG_STATUS_STRIDE_MISMATCH,
    LG_STATUS_EXPR_OVERFLOW,
    LG_STATUS_NOT_FOUND,
    LG_STATUS_NULL_POINTER,
    LG_STATUS_UNSUPPORTED_OPCODE,
    LG_STATUS_OUT_OF_MEMORY,
    LG_STATUS_OUT_OF_BOUNDS,
    LG_STATUS_HARDWARE_FAULT,
    LG_STATUS_UNEXPECTED_NAN
};

enum lg_layout {
    LG_LAYOUT_ROW_MAJOR,
    LG_LAYOUT_COL_MAJOR,
};

/// Define a tensor ID using an eight-character
/// literal.
#define LG_TENSOR_ID_8 LG_HASH_LITERAL_8

/// Tensor shape descriptor
struct lg_desc {
    /// The rank of the tensor.
    /// Must be less than LG_MAX_RANK.
    size_t rank;

    /// The dimensionality of the tensor.
    size_t dim[LG_MAX_RANK];

    /// The strides of the tensor.
    /// The order of this array must match that of `dim`.
    size_t strides[LG_MAX_RANK];
};

/// Tracks the coordinates of LG_N_TRACKED_TENSORS tensors.
///
/// All tensors in a single iter must be both broadcasted and
/// have their dims sorted in descending order.
struct lg_nditer {
    size_t coords[LG_MAX_RANK];
    struct lg_desc descs[LG_N_TRACKED_TENSORS];
    size_t indices[LG_N_TRACKED_TENSORS];
    size_t n_tracked_dims;
};

/// Increment the coordinate `axis` on `iter` and update offsets.
/// 
/// Does not perform any bounds checking.
bool LG_NDiterIncrement(struct lg_nditer *iter, size_t axis);

/// Recomputes the indices in `iter` according to its `coords`.
///
/// If you want to "jump" to a specific coordinate in a tensor, this is the
/// easiest way to do it.
void LG_NDiterGoto(struct lg_nditer *iter, size_t *coords);

/// Broadcast tensor views.
///
enum lg_status LG_ComputeBroadcastedAxes(struct lg_desc **descs, size_t n_descs);

/// Contracts the dimensions of `y`, inferring the contracted dimensions.
///
/// The contracted dimensions must be aligned at the beginning of `x0`, and `x1` with batch dimensions
/// following.
enum lg_status LG_ComputeContractedAxes(struct lg_desc *y, struct lg_desc *x0, struct lg_desc *x1, size_t n_batch_axes);

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
enum lg_status LG_SortAxes(struct lg_desc **descs, size_t n_descs);

/// Coalesce tensor axes to be as flat as possible.
///
/// Inputs to this function MUST be broadcasted AND sorted from least to greatest
/// using `lg_SortAxes`.
enum lg_status LG_CoalesceAxes(struct lg_desc **descs, size_t n_descs);

/// Compute the size in bytes of a tensor's data buffer.
size_t LG_DescSizeInBytes(struct lg_desc desc);

/// Copy a vector value to the dim `copy_to_dim`.
void LG_CopyVectorToAxis(struct lg_desc desc, lg_scalar *restrict dest, const lg_scalar *vector, size_t copy_to_axis);

/// Lays out a tensor with pre-populated `dim` and `rank` with the strides to be stored in
/// the order in `layout`. In this layout, the rightmost dimension has the unit stride.
///
/// Rows (the unit stride dimension) are padded to align with `unit_align` if `unit_align` > 1.
///
/// Does not allocate any memory; that can be done with `lg_alloc_tensor`.
///
/// This is the recommended and standard way to initialize a tensor layout.
enum lg_status LG_DescComputeLayoutStrides(struct lg_desc *desc, enum lg_layout layout, size_t unit_align);

/// Returns true if a tensor is isotropic.
/// 
/// Tensors with a rank of zero, and all scalars are considered isotropic,
/// while all vectors are considered anisotropic.
bool LG_DescIsIsotropic(struct lg_desc desc);

#endif // LG_CORE_H_
