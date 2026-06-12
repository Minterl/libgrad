#include <libgrad/core.h>

#ifndef LG_CPU_H_
#define LG_CPU_H_

void lg_backend_cpu_init(lg_backend *backend);
lg_status lg_cpu_add(void *ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION
#undef LG_CPU_IMPLEMENTATION

void lg_backend_cpu_init(lg_backend *backend) {
    *backend = (lg_backend){
        .ctx = NULL,
        .forward_vtable = {
            [LG_OPCODE_ADD] = lg_cpu_add,
        },
        .backward_vtable = {0},
    };
}

lg_status lg_cpu_add(
    void *ctx,
    lg_tensor out,
    const lg_tensor a,
    const lg_tensor b
) {
    (void)ctx;
    (void)out;
    (void)a;
    (void)b;
    return LG_STATUS_OK;
}

#endif // LG_CPU_IMPLEMENTATION
