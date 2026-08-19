#include <libgrad/internal/base.h>

#define MRV_TOKEN_STREAM_BLOCK_CAPACITY 1024
#define MRV_MAX_ERR_LEN 1024


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// error reporting etc.
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_Span {
    size_t start_offset;
    size_t end_offset;
} MRV_Span;

typedef struct
MRV_Error {
    bool      is_err;

    MRV_Span  span;

    size_t    msg_len;
    uint8_t   msg[MRV_MAX_ERR_LEN];
} MRV_Error;

size_t 
mrv_report_error_write(void *ctx, lg_str8 str) {
    MRV_Error *err = ctx;
    size_t bytes_written = lg_strcpy((lg_str8){
        .len = MRV_MAX_ERR_LEN - err->msg_len,
        .p = err->msg + err->msg_len,
    }, str);
    err->msg_len += bytes_written;
    return bytes_written;
}

void
mrv_report_error(MRV_Error *err, MRV_Span span, lg_str8 fmt, ...) {
    if (err->is_err) {
        return;
    }

    err->msg_len = 0;
    err->is_err = 1;
    err->span = span;
    
    LG_Writer w = {
        .ctx = (void*)err,
        .write = mrv_report_error_write,
    };

    va_list ap;
    va_start(ap, fmt);
    LG_StatusKind vprintf_status = lg_vprintf(&w, fmt, ap);
    (void)vprintf_status;
    va_end(ap);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// fundamental lexing data structures
///
////////////////////////////////////////////////////////////////////////////////

#define MRV_DEFINE_TOKEN_KINDS \
    MRV_X(Error,              "") \
    MRV_X(EOF,                "") \
    MRV_X(OpenParen,          "(") \
    MRV_X(CloseParen,         ")") \
    MRV_X(OpenBrace,          "{") \
    MRV_X(CloseBrace,         "}") \
    MRV_X(Colon,              ":") \
    MRV_X(Comma,              ",") \
    MRV_X(Semicolon,          ";") \
    MRV_X(Equals,             "=") \
    MRV_X(Stencil,            "stencil") \
    MRV_X(Ident,              "") \
    MRV_X(SymbolIdent,        "")

typedef int8_t 
MRV_TokenKind;
enum {
#   define MRV_X(kind, ...) MRV_TokenKind_##kind,
    MRV_DEFINE_TOKEN_KINDS
#   undef  MRV_X
};

enum {
#   define MRV_X(...) + 1
    MRV_TokenKind_COUNT = 0 MRV_DEFINE_TOKEN_KINDS
#   undef MRV_X
};

static const struct {
    lg_str8 string; 
    lg_str8 kind_string;
    uint32_t hash; 
} 
MRV_TOKEN_TABLE[MRV_TokenKind_COUNT] = {
#   define MRV_X(kind, str) [MRV_TokenKind_##kind] = { \
        .string = lg_str8_lit(str), \
        .kind_string = lg_str8_lit(#kind), \
        .hash = lg_hash_lit_16(str) \
    },
    MRV_DEFINE_TOKEN_KINDS
#   undef MRV_X
};

#define mrv_token_kind_is_keyword(kind) (MRV_TOKEN_TABLE[kind].string.len > 1)
#define mrv_token_as_str(kind) (MRV_TOKEN_TABLE[kind].kind_string)
#define mrv_token_assoc_str(kind) (MRV_TOKEN_TABLE[kind].string);

typedef struct
MRV_Token {
    MRV_TokenKind kind;
    MRV_Span span;
} MRV_Token;

typedef struct
MRV_TokenStreamBlock {
    struct MRV_TokenStreamBlock *next;
    struct MRV_TokenStreamBlock *prev;

    MRV_Token  tokens[MRV_TOKEN_STREAM_BLOCK_CAPACITY];
} MRV_TokenStreamBlock;

typedef struct 
MRV_TokenStream {
    size_t tail_len;
    MRV_TokenStreamBlock *tail;
} MRV_TokenStream;

typedef struct
MRV_LexerContext {
    lg_str8        text;
    size_t         current_offset;

    LG_Allocator  *artifact_allocator;

    MRV_Error      err;
} MRV_LexerContext;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// private lexer utils
///
////////////////////////////////////////////////////////////////////////////////

#define mrv_span_zero_len(offset) (MRV_Span){ .start_offset = (offset), .end_offset = (offset) }

void
mrv_tstream_append(MRV_TokenStream *tstream, LG_Allocator *artifact_allocator, MRV_Token tok) {
    if (lg_likely(
        tstream->tail != NULL &&
        tstream->tail_len < MRV_TOKEN_STREAM_BLOCK_CAPACITY
    )) {
        lg_memcpy(tstream->tail->tokens + tstream->tail_len, &tok, sizeof(MRV_Token));
        return;
    }

    MRV_TokenStreamBlock *next_block = (MRV_TokenStreamBlock*)lg_alloc_zero(artifact_allocator, sizeof(MRV_TokenStream));
    lg_assert(next_block != NULL);

    tstream->tail->next = next_block;
    next_block->prev = tstream->tail;
    tstream->tail = next_block;
    tstream->tail_len = 0;

    mrv_tstream_append(tstream, artifact_allocator, tok);
}

lg_force_inline lg_str8
mrv_span_to_str8(MRV_Span span, lg_str8 text) {
    return (lg_str8){ .len = span.end_offset - span.start_offset, .p = text.p + span.start_offset };
}

lg_force_inline uint8_t
mrv_lexer_peek(MRV_LexerContext *ctx) {
    if (lg_likely(ctx->current_offset < ctx->text.len - 1)) {
        lg_assert(ctx->text.p[ctx->current_offset + 1] != '\0');
        return ctx->text.p[ctx->current_offset + 1];
    }
    return '\0';
}

lg_force_inline void
mrv_lexer_skip(MRV_LexerContext *ctx) {
    if (lg_likely(ctx->current_offset < ctx->text.len - 1)) {
        ctx->current_offset++;
    }
}

lg_force_inline void
mrv_lexer_skip_whitespace(MRV_LexerContext *ctx) {
    for (
        uint8_t ch_i = ctx->text.p[ctx->current_offset];
        lg_char_is_whitespace(ch_i);
        ctx->current_offset++, ch_i = ctx->text.p[ctx->current_offset]
    );
}

lg_force_inline MRV_Token
mrv_lexer_consume_char(MRV_LexerContext *ctx) {
    if (ctx->current_offset >= ctx->text.len) {
        mrv_report_error(&ctx->err, mrv_span_zero_len(ctx->text.len), lg_str8_lit("unexpected EOF"));
        return (MRV_Token){ .kind = MRV_TokenKind_Error };
    }

    uint8_t ch = ctx->text.p[ctx->current_offset];

    MRV_TokenKind kind = MRV_TokenKind_Error;
    for (uint8_t i = 0; i < MRV_TokenKind_COUNT; i++) {
        if (MRV_TOKEN_TABLE[i].string.len != 1) {
            continue;
        }

        if(MRV_TOKEN_TABLE[i].string.p[0] == ch) {
            kind = i;
            break;
        }
    }

    MRV_Token tok = {
        .kind = kind,
        .span.start_offset = ctx->current_offset,  
        .span.end_offset = ctx->current_offset + 1,  
    };

    ctx->current_offset++;

    return tok;
}

lg_force_inline MRV_Token
mrv_lexer_scan_ident(MRV_LexerContext *ctx, MRV_TokenKind expected_kind) {
    MRV_Token tok = { 
        .kind = expected_kind,
        .span.start_offset = ctx->current_offset,
    };

    /////////////////////////////////////////////////////
    /// ~~ scan sequence ~~

    bool is_first = true;
    while (ctx->current_offset < ctx->text.len) {
        uint8_t ch_i = ctx->text.p[ctx->current_offset];

        ctx->current_offset++;
        tok.span.end_offset = ctx->current_offset;

        if (lg_unlikely(ctx->current_offset >= ctx->text.len)) {
            mrv_report_error(&ctx->err, tok.span, lg_str8_lit("unexpected EOF"));
            return (MRV_Token){ .kind = MRV_TokenKind_Error };
        } 
        if (lg_unlikely(is_first && expected_kind == MRV_TokenKind_Ident)) {
            mrv_report_error(&ctx->err, tok.span, lg_str8_lit("expected letter, found number"));
            return (MRV_Token){ .kind = MRV_TokenKind_Error };
        }
        if (lg_unlikely(!lg_char_is_alphanumeric(ch_i))) {
            break;
        }

        is_first = false;
    }

    if (tok.span.start_offset == tok.span.end_offset) {
        mrv_report_error(&ctx->err, tok.span, lg_str8_lit("expected alphanumeric sequence"));
        return (MRV_Token){ .kind = MRV_TokenKind_Error };
    }
    

    /////////////////////////////////////////////////////
    /// ~~ scan for keywords ~~

    lg_str8 ident_string = mrv_span_to_str8(tok.span, ctx->text);
    uint32_t ident_hash = lg_hash_16(ident_string.p, ident_string.len);

    for (uint8_t kind = 0; kind < MRV_TokenKind_COUNT; kind++) {
        if (!mrv_token_kind_is_keyword(kind)) {
            continue; 
        }
        if (ident_hash == MRV_TOKEN_TABLE[kind].hash) {
            tok.kind = kind;
            break;
        }
    }

    return tok;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// the part that does the lexing
///
////////////////////////////////////////////////////////////////////////////////

void
mrv_lex(MRV_LexerContext *ctx, MRV_TokenStream *tstream) {
    while (ctx->current_offset < ctx->text.len) {
        mrv_lexer_skip_whitespace(ctx);

        switch (ctx->text.p[ctx->current_offset]) {
        case 'A'...'Z':
        case 'a'...'z': {
            MRV_Token sym_ident = mrv_lexer_scan_ident(ctx, MRV_TokenKind_Ident);
            mrv_tstream_append(tstream, ctx->artifact_allocator, sym_ident);

            break; 
        }

        case '%': {
            mrv_lexer_skip(ctx);

            MRV_Token sym_ident = mrv_lexer_scan_ident(ctx, MRV_TokenKind_SymbolIdent);
            mrv_tstream_append(tstream, ctx->artifact_allocator, sym_ident);

            break;
        }

        case '(':
        case ')':
        case '{':
        case '}':
        case ':':
        case ',':
        case ';':
        case '=': {
            MRV_Token ch = mrv_lexer_consume_char(ctx);
            mrv_tstream_append(tstream, ctx->artifact_allocator, ch);

            break;
        }

        default: {
            MRV_Span err_span = mrv_span_zero_len(ctx->current_offset);
            lg_str8 unexpected_char = mrv_span_to_str8(err_span, ctx->text);

            mrv_tstream_append(tstream, ctx->artifact_allocator, (MRV_Token){
                .kind = MRV_TokenKind_Error,
                .span = err_span,
            });
            mrv_report_error(&ctx->err, err_span, lg_str8_lit("unexpected character %{str}"), unexpected_char);

            mrv_lexer_skip(ctx);
        }
        }
    }

    mrv_tstream_append(tstream, ctx->artifact_allocator, (MRV_Token){
        .kind = MRV_TokenKind_EOF,
        .span = mrv_span_zero_len(ctx->current_offset),
    });
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// fundamental parsing data structures
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_ParserContext {
    lg_str8 text;

    MRV_TokenStream       *tstream;
    MRV_TokenStreamBlock  *cur_block;
    MRV_Token             *cur_token;

    MRV_Error              err;
    LG_Allocator          *artifact_allocator;
} MRV_ParserContext;

typedef uint8_t
MRV_ParserStatusKind;
enum {
    MRV_ParserStatusKind_ErrorOccurred = -1,
    MRV_ParserStatusKind_Continue,
    MRV_ParserStatusKind_Done,
};

void
mrv_parser_begin(MRV_ParserContext *parser, MRV_TokenStream *tstream) {
    lg_assert(parser != NULL);
    lg_assert(tstream != NULL);
    lg_assert(tstream->tail->next == NULL);

    lg_memzero(parser, sizeof(MRV_ParserContext));
    parser->tstream = tstream;

    // this will loop forever if there are cycles
    MRV_TokenStreamBlock *iter_block = tstream->tail;
    while (iter_block != NULL) {
        parser->cur_block = iter_block;
        iter_block = iter_block->prev;
    }

    if (parser->cur_block != NULL) {
        parser->cur_token = parser->cur_block->tokens;
    }
}

lg_force_inline MRV_ParserStatusKind 
mrv_parser_find_next(
    MRV_ParserContext *parser,
    MRV_Token **out_next_token,
    MRV_TokenStreamBlock **out_next_block
) {
    lg_assert(parser != NULL);
    lg_assert(out_next_block != NULL);
    lg_assert(out_next_token != NULL);

    if (parser->cur_block == NULL) {
        return MRV_ParserStatusKind_Done;
    }
    if (parser->cur_token == NULL) {
        *out_next_block = parser->cur_block;
        *out_next_token = parser->cur_block->tokens;
        return MRV_ParserStatusKind_Continue;
    }

    const ptrdiff_t cur_offset = parser->cur_token - parser->cur_block->tokens;
    lg_assert(cur_offset >= 0);

    const size_t cur_len = parser->cur_block == parser->tstream->tail ? 
        parser->tstream->tail_len :
        MRV_TOKEN_STREAM_BLOCK_CAPACITY;

    if (lg_likely((size_t)cur_offset < cur_len)) {
        *out_next_block = parser->cur_block;
        *out_next_token = parser->cur_token + 1;
        return MRV_ParserStatusKind_Continue;
    }

    if (parser->cur_block->next == NULL) {
        return MRV_ParserStatusKind_Done;
    }

    *out_next_block = parser->cur_block->next;
    *out_next_token = parser->cur_block->next->tokens;

    return true;
}

MRV_ParserStatusKind
mrv_parser_expect_consume(MRV_ParserContext *parser, MRV_TokenKind expected_kind, MRV_Token *out_token) {
    lg_assert(parser != NULL);
    lg_assert(out_token != NULL);
    lg_assert(expected_kind != MRV_TokenKind_Error);

    MRV_Token *next_token;
    MRV_TokenStreamBlock *next_block;
    bool has_next = mrv_parser_find_next(parser, &next_token, &next_block);
    lg_assert(has_next); // this function shouldn't be called where there isn't a next token
                         // e.g don't expect a token after an EOF

    if (expected_kind != next_token->kind) {
        mrv_report_error(
            &parser->err,
            next_token->span,
            lg_str8_lit("expected %{str}, found %{str}"), 
            mrv_token_as_str(expected_kind),    
            mrv_span_to_str8(next_token->span, parser->text)
        );
        return MRV_ParserStatusKind_ErrorOccurred;        
    }

    parser->cur_token = next_token;
    parser->cur_block = next_block;

    *out_token = *next_token;

    return MRV_ParserStatusKind_Continue;
}

MRV_ParserStatusKind
mrv_parser_peek(MRV_ParserContext *parser, MRV_Token *out_token) {
    lg_assert(out_token != NULL);

    MRV_Token *next_token;
    MRV_TokenStreamBlock *next_block;
    bool has_next = mrv_parser_find_next(parser, &next_token, &next_block);
    if (!has_next) {
        return MRV_ParserStatusKind_Done;
    }

    *out_token = *next_token;

    return MRV_ParserStatusKind_Continue;
}
