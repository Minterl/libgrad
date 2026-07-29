// TODO: once the rest of the lib builds, just use the main header
#include <libgrad/internal/strings.h>

#ifndef TEST_IMPLEMENTATION
#define TEST_IMPLEMENTATION
#endif // TEST_IMPLEMENTATION
#include "testing.h"

#include <stdio.h>

size_t WriteToStdout(void *ctx, const struct lg_string str) {
    (void)ctx;
    size_t i = 0;
    for (; i < str.len; i++) {
        putchar(str.p[i]);
    }
    return i;
}

static struct lg_writer stdout_writer = {
    .Write = WriteToStdout,
};

test_status test_printf() {
    test_assert(LG_Printf(&stdout_writer, LG_STRING_LITERAL("asdf: %{i64}\n"), 13) == LG_STATUS_OK, "int failed to print");
    test_assert(LG_Printf(&stdout_writer, LG_STRING_LITERAL("asdf: %{str}\n"), LG_STRING_LITERAL("asdfasdf")) == LG_STATUS_OK, "string failed to print");
    test_assert(LG_Printf(&stdout_writer, LG_STRING_LITERAL("asdf: %{cstr}\n"), "asdfasdf") == LG_STATUS_OK, "cstring failed to print");
    test_assert(LG_Printf(&stdout_writer, LG_STRING_LITERAL("asdf: %{cstr\n"), "asdfasdf") == LG_STATUS_INVALID_ARGUMENT, "unterminated fmtspec didn't fail");
    test_assert(
        LG_Printf(&stdout_writer, LG_STRING_LITERAL("asdf: %{i64} %{cstr}\n"), 14, "asdfasdf") == LG_STATUS_OK,
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
