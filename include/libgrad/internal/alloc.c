#include <libgrad/internal/vm.h>
#include <libgrad/internal/alloc.h>

uint8_t *LG__AllocZero(struct lg_allocator *alloc, size_t size_bytes) {
    uint8_t *ptr = alloc->Alloc(alloc->ctx, size_bytes);
    if (ptr == NULL) {
        return ptr;
    }
    LG__ZERO(ptr, size_bytes);
    return ptr;
}

void LG__Free(struct lg_allocator *alloc, void *ptr) {
    if (alloc->Free) {
        alloc->Free(alloc->ctx, ptr);
    }
}

enum lg_status LG__AllocContiguousBlocks(
    struct lg_allocator *alloc,
    uint8_t **out_ptrs,
    size_t *LG_NULLABLE out_bytes_allocated,
    const size_t *sizes,
    size_t n,
    size_t align
) {
    size_t size = 0;
    for (size_t i = 0; i < n; i++) {
        size += LG__ALIGN_UP(sizes[i], align);
    }

    uint8_t *ptr = LG__AllocZero(alloc, size);
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

struct lg_scratch_node *LG__AcquireScratch(struct lg_allocator *alloc) {
    if (LG_ALLOCATOR_SUPPORTS_SCRATCH(alloc)) {
        return alloc->AcquireScratch(alloc->ctx);
    }
    return NULL;
}

uint8_t *LG__AllocScratch(struct lg_allocator *alloc, struct lg_scratch_node **waypoint, size_t size_bytes) {
    if (LG_ALLOCATOR_SUPPORTS_SCRATCH(alloc)) {
        return LG__AllocZero(alloc, size_bytes);
    }
    
    const size_t size_total = sizeof(struct lg_scratch_node) + size_bytes;
    struct lg_scratch_node *new_head = (struct lg_scratch_node*)LG__AllocZero(alloc, size_total);
    if (new_head == NULL) {
        return NULL;
    }
    uint8_t *buf_ptr = (uint8_t*)(new_head + 1);

    new_head->next = *waypoint;
    *waypoint = new_head;

    return buf_ptr;
}

void LG__ReleaseScratch(struct lg_allocator *alloc, struct lg_scratch_node **waypoint) {
    if (LG_ALLOCATOR_SUPPORTS_SCRATCH(alloc)) {
        return alloc->ReleaseScratch(alloc->ctx, (void*)waypoint);
    }
    if (*waypoint == NULL) {
        return;
    }

    struct lg_scratch_node *node = *waypoint;
    while (node != NULL) {
        struct lg_scratch_node *next = node->next;
        LG__Free(alloc, node);
        node = next;
    }

    *waypoint = NULL;
}
