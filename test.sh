#!/usr/bin/env bash
set -euo pipefail

mkdir -p out

bear -- cc -o out/test-core.out \
    -Wall -Wextra -g -std=c99 \
    -Iinclude \
    -Itest \
    -DLG_SAFE \
    -DLG_DEBUG \
    test/core.c

./out/test-core.out
