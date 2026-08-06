#define LG_FNV_PRIME 16777619U
#define LG_FNV_OFFSET_BASIS 2166136261U

#define lg_fnv_getc(str, idx, len) ((idx) < (len) ? (str)[idx] : '\0')
#define lg_fnv_step(hash, c) (((hash) ^ (char)(c)) * LG_FNV_PRIME)

#define lg_hash_lit_16(str) lg_hash_16((str), ((size_t)(sizeof(str))))
#define lg_hash_16(str, len) \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step( \
    lg_fnv_step(LG_FNV_OFFSET_BASIS, lg_fnv_getc(str, 0, len)), \
    lg_fnv_getc(str, 1, len)), \
    lg_fnv_getc(str, 2, len)), \
    lg_fnv_getc(str, 3, len)), \
    lg_fnv_getc(str, 4, len)), \
    lg_fnv_getc(str, 5, len)), \
    lg_fnv_getc(str, 6, len)), \
    lg_fnv_getc(str, 7, len)), \
    lg_fnv_getc(str, 8, len)), \
    lg_fnv_getc(str, 9, len)), \
    lg_fnv_getc(str, 10, len)), \
    lg_fnv_getc(str, 11, len)), \
    lg_fnv_getc(str, 12, len)), \
    lg_fnv_getc(str, 13, len)), \
    lg_fnv_getc(str, 14, len)), \
    lg_fnv_getc(str, 15, len))
