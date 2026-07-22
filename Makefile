CFLAGS = \
	-Wall -Werror -Wextra -g -std=c99 \
	-Iinclude \
	-Itest \
	-DLG_SAFE \
	-DLG_DEBUG

.PHONY: test
test:
	mkdir -p out && \
	cc -o out/test-core.out \
		$(CFLAGS) \
		-fsanitize=address \
		-fsanitize=bounds \
		test/core.c

.PHONY: examples/mnist
examples/mnist:
	mkdir -p out/examples && \
	cc -o out/examples/mnist.out \
		$(CFLAGS) \
		-Iexamples/include \
		-fsanitize=address \
		-fsanitize=bounds \
		examples/mnist/main.c
