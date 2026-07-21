#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

enum lg_status LG__AllocContiguousBlocks(
    struct lg_allocator *alloc,
    uint8_t **out_ptrs,
    size_t *out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        size += LG__ALIGN_UP(sizes[i], align);
    }

    uint8_t *ptr = alloc->Alloc(alloc->ctx, size);
    if (ptr == NULL) {
        return LG_STATUS_OUT_OF_MEMORY;
    }

    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < n; i++) {
        out_ptrs[i] = ptr + current_offset;
        current_offset += LG__ALIGN_UP(sizes[i], align);
    }

    return LG_STATUS_OK;
}
