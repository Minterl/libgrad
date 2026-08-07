// TODO: once the rest of the lib builds, just use the main header
#include "libgrad/internal/core.h"
#include <libgrad/internal/strings.h>

#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#include "testing.h"

#include <stdio.h>

size_t write_to_stdout(void *ctx, const lg_str8 str) {
    (void)ctx;
    size_t i = 0;
    for (; i < str.len; i++) {
        putchar(str.p[i]);
    }
    return i;
}

static lg_writer stdout_writer = {
    .write = write_to_stdout,
};

test_status test_printf() {
    test_assert(lg_printf(&stdout_writer, lg_str8_lit("asdf: %{i64}\n"), 13) == LG_StatusKind_OK, "int failed to print");
    test_assert(lg_printf(&stdout_writer, lg_str8_lit("asdf: %{str}\n"), lg_str8_lit("asdfasdf")) == LG_StatusKind_OK, "string failed to print");
    test_assert(lg_printf(&stdout_writer, lg_str8_lit("asdf: %{cstr}\n"), "asdfasdf") == LG_StatusKind_OK, "cstring failed to print");
    test_assert(lg_printf(&stdout_writer, lg_str8_lit("asdf: %{cstr\n"), "asdfasdf") == LG_StatusKind_InvalidArgument, "unterminated fmtspec didn't fail");
    test_assert(
        lg_printf(&stdout_writer, lg_str8_lit("asdf: %{i64} %{cstr}\n"), 14, "asdfasdf") == LG_StatusKind_OK,
        "mixed failed to print"
    );
    return TEST_STATUS_OK;
}

int main() {
    test_run(printf);
    return 0;
}

#include <libgrad/internal/strings.c>
#include <libgrad/internal/debug.c>
