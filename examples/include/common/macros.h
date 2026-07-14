#ifndef MACROS_H_
#define MACROS_H_

#define FAILF(...) do { \
    printf("failed at %s:%d -- ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); \
    printf("\n"); \
} while (0) \

#endif // MACROS_H_
