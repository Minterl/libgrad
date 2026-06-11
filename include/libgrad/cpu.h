#include <libgrad/core.h>

#ifndef LG_CPU_H_
#define LG_CPU_H_

typedef struct lg_backend_cpu {
    
} lg_backend_cpu;

lg_backend lg_backend_cpu_interface(lg_backend_cpu *backend);

lg_status lg_cpu_add(void *ctx, lg_tensor out, const lg_tensor a, const lg_tensor b);

#endif // LG_CPU_H_


#ifdef LG_CPU_IMPLEMENTATION_
#undef LG_CPU_IMPLEMENTATION_

lg_backend lg_backend_cpu_interface(lg_backend_cpu *backend) {
    return (lg_backend){
        .ctx = (void*)backend,
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

#endif // LG_CPU_IMPLEMENTATION_
