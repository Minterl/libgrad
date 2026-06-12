#ifndef LG_CORE_IMPLEMENTATION
#define LG_CORE_IMPLEMENTATION
#endif // LG_CORE_IMPLEMENTATION
#ifndef LG_CPU_IMPLEMENTATION
#define LG_CPU_IMPLEMENTATION
#endif // LG_CPU_IMPLEMENTATION
#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif //TEST_IMPLEMENTATION
 
#include <stdbool.h>
#include <libgrad/core.h>
#include <libgrad/cpu.h>
#include "testing.h"

test_status test_cpu_add() {
    lg_backend_cpu cpu_backend = {};
    lg_backend backend = lg_backend_cpu_interface(&cpu_backend);
    lg_context ctx = (lg_context){
        .backend = &backend,
        .tape = NULL,
    };

    lg_tensor a = (lg_tensor){
        1, 
        {1},
        {1},
        NULL,
        NULL,
    };
    lg_tensor b = (lg_tensor){
        1, 
        {1},
        {1},
        NULL,
        NULL,
    };
    lg_tensor out = (lg_tensor){
        1, 
        {1},
        {1},
        NULL,
        NULL,
    };
    test_assert(lg_add(ctx, out, a, b) == LG_STATUS_OK, "should not fail %s", "afds");

    return TEST_STATUS_OK;
}

int main(void) {
    test_run(cpu_add);
    return 0;
}
