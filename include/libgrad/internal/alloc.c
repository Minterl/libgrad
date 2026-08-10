#include <libgrad/internal/alloc.h>
#include <libgrad/internal/debug.h>

uint8_t*
lg_alloc_zero(LG_Allocator *alloc, size_t size_bytes) {
    uint8_t *ptr = alloc->alloc(alloc->ctx, size_bytes);
    if (ptr == NULL) {
        return ptr;
    }

    if (!(alloc->flags & LG_AllocatorFlag_AssumeZeroed)) {
        lg_memzero(ptr, size_bytes);
    }

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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// 
/// Slab utilities
///
////////////////////////////////////////////////////////////////////////////////

lg_force_inline void
lg_slab_unlink(LG_Slab *slab) {
    if (slab->prev != NULL) {
        slab->prev->next = slab->next;
    }            
    if (slab->next != NULL) {
        slab->next->prev = slab->prev;
    }            
    slab->prev = NULL;
    slab->next = NULL;
}

lg_force_inline void
lg_slab_append(LG_Slab *slab, LG_Slab *onto) {
    while (true) {
        lg_assert(onto != slab);
        if (onto->next == NULL) {
            break;
        }
        onto = onto->next;
    }

    onto->next = slab;
    slab->prev = onto;
}

lg_force_inline void
lg_slab_free_from(LG_Slab *slab, LG_Allocator *alloc) {
    LG_Slab *next = slab;
    while (next != NULL) {
        LG_Slab *current = next;
        next = current->next;
        lg_free(alloc, current);
    }
}

lg_force_inline LG_Slab*
lg_slab_find_first(LG_Slab *slab) {
    bool is_first_iteration = true;
    LG_Slab *first = slab;
    while (true) {
        lg_assert(is_first_iteration == true || first != slab);
        is_first_iteration = false;
        if (first->prev == NULL) {
            break;
        }
        first = first->prev;
    }
    return first;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// 
/// Public arena APIs
///
////////////////////////////////////////////////////////////////////////////////

void
lg_arena_init(LG_Arena *arena, LG_Allocator *host) {
    lg_memzero(arena, sizeof(LG_Arena));
    arena->host = host;
}

uint8_t*
lg_arena_alloc(LG_Arena *arena, size_t unaligned_size_bytes, size_t align) {
    const size_t size_bytes = lg_align_up(unaligned_size_bytes, align);


    ////////////////////////////////////////
    // ~~ Plan A: use the current slab ~~

    if (lg_likely(
        arena->current_slab != NULL &&
        arena->current_offset + size_bytes <= arena->current_slab->cap
    )) {
        lg_memzero(arena->current_slab->buf + arena->current_offset, size_bytes);

        const size_t prev_offset = arena->current_offset;
        arena->current_offset += size_bytes;

        return arena->current_slab->buf + prev_offset;
    }


    //////////////////////////////////////////////////
    // ~~ Plan B: first-fit an existing free slab ~~

    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        goto plan_c;
    }

    lg_assert(arena->recycled_slabs_head == NULL || arena->recycled_slabs_head->prev == NULL);

    LG_Slab *to_reuse = arena->recycled_slabs_head;
    while (to_reuse != NULL) {
        if (to_reuse->cap >= size_bytes) {
            lg_slab_unlink(to_reuse);

            if (arena->current_slab != NULL) {
                arena->current_slab->next = to_reuse;
                to_reuse->prev = arena->current_slab;
            }

            arena->current_slab = to_reuse;
            arena->current_offset = size_bytes;

            lg_memzero(to_reuse->buf, size_bytes);

            return to_reuse->buf;
        }

        to_reuse = to_reuse->next;
    }


    ////////////////////////////////////////
    // ~~ Plan C: allocate a new slab ~~

plan_c:;

    const size_t buf_size = size_bytes > arena->host->default_slab_size_bytes ?
        size_bytes :
        arena->host->default_slab_size_bytes;
    const size_t total_size = sizeof(LG_Slab) + buf_size;

    LG_Slab *next = arena->host->alloc(arena->host->ctx, total_size);
    if (next == NULL) {
        return NULL;
    }

    lg_memzero(next, sizeof(LG_Slab));
    next->cap = buf_size;

    if (arena->current_slab != NULL) {
        lg_assert(arena->current_slab->next == NULL);
        arena->current_slab->next = next;
        next->prev = arena->current_slab;
    }

    arena->current_slab = next;
    arena->current_offset = size_bytes;

    lg_memzero(next->buf, size_bytes);

    return next->buf;
}

LG_Scope
lg_push_scope(LG_Arena *arena) {
    return (LG_Scope){
        .offset = arena->current_offset,
        .slab = arena->current_slab,
    };
}

void
lg_pop_scope(LG_Arena *arena, LG_Scope scope) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        if (scope.slab == arena->current_slab) {
            arena->current_slab = NULL;
        }

        lg_slab_free_from(scope.slab, arena->host);

        return;
    }

    LG_Slab *to_recycle = scope.offset == 0 ?
        scope.slab :
        scope.slab->next;

    if (to_recycle == NULL) {
        return;
    }
    if (to_recycle == arena->current_slab) {
        arena->current_slab = NULL;
    }
    if (to_recycle->prev != NULL) {
        to_recycle->prev->next = NULL;
    }

    lg_slab_append(to_recycle, arena->recycled_slabs_head);

    arena->current_offset = scope.offset;
}

void
lg_arena_free_recycled(LG_Arena *arena) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        lg_assert(arena->recycled_slabs_head == NULL);
        return;
    }
    if (arena->recycled_slabs_head == NULL) {
        return;
    }

    lg_assert(arena->recycled_slabs_head->prev == NULL);
    lg_slab_free_from(arena->recycled_slabs_head, arena->host);
    arena->recycled_slabs_head = NULL;
}

void
lg_arena_recycle_all(LG_Arena *arena) {
    if (arena->host->flags & LG_AllocatorFlag_NoRecycle) {
        return;
    }

    LG_Slab *first_active = lg_slab_find_first(arena->current_slab);
    lg_slab_append(first_active, arena->recycled_slabs_head);
    arena->current_slab = NULL;
}

void
lg_arena_free_all(LG_Arena *arena) {
    LG_Slab *first_active = lg_slab_find_first(arena->current_slab);
    lg_slab_free_from(first_active, arena->host);
    lg_arena_free_recycled(arena);
    arena->current_offset = 0;
    arena->current_slab = NULL;
}
