#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

uint8_t*
lg_alloc_zero(LG_Allocator *alloc, size_t size_bytes) {
    uint8_t *ptr = alloc->alloc(alloc->ctx, size_bytes);
    if (ptr == NULL) {
        return ptr;
    }
    lg_memzero(ptr, size_bytes);
    return ptr;
}

void 
lg_free(LG_Allocator *alloc, void *ptr) {
    if (alloc->free) {
        alloc->free(alloc->ctx, ptr);
    }
}

LG_StatusKind 
lg_alloc_contiguous_blocks(
    LG_Allocator *alloc,
    uint8_t **out_ptrs,
    size_t *lg_nullable out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        size += lg_align_up(sizes[i], align);
    }

    uint8_t *ptr = lg_alloc_zero(alloc, size);
    if (ptr == NULL) {
        return LG_StatusKind_OutOfMemory;
    }

    if (out_bytes_allocated != NULL) {
        *out_bytes_allocated = size;
    }

    size_t current_offset = 0;
    for (size_t i = 0; i < n; i++) {
        out_ptrs[i] = ptr + current_offset;
        current_offset += lg_align_up(sizes[i], align);
    }

    return LG_StatusKind_OK;
}

LG_ScratchWaypoint*
lg_scratch_acquire(LG_Allocator *alloc) {
    if (lg_allocator_supports_scratch(alloc)) {
        return alloc->scratch_acquire(alloc->ctx);
    }
    return NULL;
}

uint8_t*
lg_scratch_alloc(LG_Allocator *alloc, LG_ScratchWaypoint **waypoint, size_t size_bytes) {
    if (lg_allocator_supports_scratch(alloc)) {
        return lg_alloc_zero(alloc, size_bytes);
    }
    
    const size_t size_total = sizeof(LG_ScratchWaypoint) + size_bytes;
    LG_ScratchWaypoint *new_head = (LG_ScratchWaypoint*)lg_alloc_zero(alloc, size_total);
    if (new_head == NULL) {
        return NULL;
    }
    uint8_t *buf_ptr = (uint8_t*)(new_head + 1);

    new_head->next = *waypoint;
    *waypoint = new_head;

    return buf_ptr;
}

void 
lg_scratch_release(LG_Allocator *alloc, LG_ScratchWaypoint **waypoint) {
    if (lg_allocator_supports_scratch(alloc)) {
        return alloc->scratch_release(alloc->ctx, (void*)waypoint);
    }
    if (*waypoint == NULL) {
        return;
    }

    LG_ScratchWaypoint *node = *waypoint;
    while (node != NULL) {
        LG_ScratchWaypoint *next = node->next;
        lg_free(alloc, node);
        node = next;
    }

    *waypoint = NULL;
}
