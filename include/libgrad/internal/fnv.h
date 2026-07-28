#define LG_FNV_PRIME 16777619U
#define LG_FNV_OFFSET_BASIS 2166136261U

#define LG_FNV_GETC(str, idx, len) ((idx) < (len) ? (str)[idx] : '\0')
#define LG_FNV_STEP(hash, c) (((hash) ^ (char)(c)) * LG_FNV_PRIME)

#define LG_HASH_LITERAL_16(str) LG_HASH_16((str), ((size_t)(sizeof(str))))
#define LG_HASH_16(str, len) \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP(LG_FNV_OFFSET_BASIS, LG_FNV_GETC(str, 0, len)), \
    LG_FNV_GETC(str, 1, len)), \
    LG_FNV_GETC(str, 2, len)), \
    LG_FNV_GETC(str, 3, len)), \
    LG_FNV_GETC(str, 4, len)), \
    LG_FNV_GETC(str, 5, len)), \
    LG_FNV_GETC(str, 6, len)), \
    LG_FNV_GETC(str, 7, len)), \
    LG_FNV_GETC(str, 8, len)), \
    LG_FNV_GETC(str, 9, len)), \
    LG_FNV_GETC(str, 10, len)), \
    LG_FNV_GETC(str, 11, len)), \
    LG_FNV_GETC(str, 12, len)), \
    LG_FNV_GETC(str, 13, len)), \
    LG_FNV_GETC(str, 14, len)), \
    LG_FNV_GETC(str, 15, len))
