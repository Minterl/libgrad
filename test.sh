#!/usr/bin/env bash

set -euxo pipefail

mkdir -p out

bear -- cc -o out/test-core.out \
    -Wall -Wextra -g -std=c99 \
    -Iinclude \
    -DLG_SAFE \
    test/core.c

./out/test-core.out
