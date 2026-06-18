#define LG_FNV_PRIME 16777619U
#define LG_FNV_OFFSET_BASIS 2166136261U

#define LG_FNV_STEP(hash, c) (((hash) ^ (char)(c)) * LG_FNV_PRIME)

#define LG_HASH_LITERAL_8(str) \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP( \
    LG_FNV_STEP(LG_FNV_OFFSET_BASIS, str[0]), \
    str[1]), \
    str[2]), \
    str[3]), \
    str[4]), \
    str[5]), \
    str[6]), \
    str[7])
