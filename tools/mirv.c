#include <libgrad/internal/base.h>

#define MRV_TOKEN_STREAM_BLOCK_CAPACITY 1024
#define MRV_MAX_ERR_LEN 1024


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// error reporting & the very important span
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_Span {
    size_t offset;
    size_t len;
} MRV_Span;

typedef struct
MRV_Error {
    bool       is_err;
    LG_Writer *writer;
    MRV_Span   span;
} MRV_Error;

void
mrv_report_error(MRV_Error *err, MRV_Span span, lg_str8 fmt, ...) {
    if (err->is_err) {
        return;
    }

    err->is_err = 1;
    err->span = span;

    lg_printf(err->writer, lg_str8_lit("at offsets %{i64}-%{i64}:\n"), span.offset, span.offset + span.len);

    va_list ap;
    va_start(ap, fmt);
    LG_StatusKind vprintf_status = lg_vprintf(err->writer, fmt, ap);
    (void)vprintf_status;
    va_end(ap);

    lg_write(err->writer, lg_str8_lit("\n"));
}

#define mrv_span_zero_len(offset_) (MRV_Span){ .offset = (offset_) }


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
    MRV_X(OpenAngleBracket,   "<") \
    MRV_X(CloseAngleBracket,  ">") \
    MRV_X(Language,           "language") \
    MRV_X(Type,               "type") \
    MRV_X(Operator,           "operator") \
    MRV_X(RightArrow,         "->") \
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

const struct {
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

    LG_Allocator  *artifact;

    MRV_Error      err;
} MRV_LexerContext;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// private lexer utils
///
////////////////////////////////////////////////////////////////////////////////

void
mrv_tstream_append(MRV_TokenStream *tstream, LG_Allocator *artifact_allocator, MRV_Token tok) {
    if (lg_likely(
        tstream->tail != NULL &&
        tstream->tail_len < MRV_TOKEN_STREAM_BLOCK_CAPACITY - 1
    )) {
        lg_memcpy(tstream->tail->tokens + tstream->tail_len, &tok, sizeof(MRV_Token));
        tstream->tail_len++;
        return;
    }

    MRV_TokenStreamBlock *next_block = (MRV_TokenStreamBlock*)lg_alloc_zero(artifact_allocator, sizeof(MRV_TokenStreamBlock));
    lg_assert(next_block != NULL);

    if (tstream->tail != NULL) {
        tstream->tail->next = next_block;
    }
    next_block->prev = tstream->tail;

    tstream->tail = next_block;
    tstream->tail_len = 0;

    mrv_tstream_append(tstream, artifact_allocator, tok);
}

void
mrv_tstream_destroy(MRV_TokenStream *tstream, LG_Allocator *artifact_allocator) {
    MRV_TokenStreamBlock *iter_block = tstream->tail;
    while (iter_block != NULL) {
        MRV_TokenStreamBlock *temp = iter_block->prev;
        lg_free(artifact_allocator, iter_block);
        iter_block = temp;
    }
    lg_memzero(tstream, sizeof(MRV_TokenStream));
}

lg_force_inline lg_str8
mrv_span_to_str8(MRV_Span span, lg_str8 text) {
    return (lg_str8){ .len = span.len, .p = text.p + span.offset };
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
    if (lg_likely(ctx->current_offset < ctx->text.len)) {
        ctx->current_offset++;
    }
}

lg_force_inline bool
mrv_lexer_match_sequence(MRV_LexerContext *ctx, lg_str8 seq) {
    if (ctx->current_offset + seq.len >= ctx->text.len) {
        return false;
    }
    lg_str8 next_n = (lg_str8){ .len = seq.len, .p = &ctx->text.p[ctx->current_offset]};
    if (lg_strcmp(next_n, seq) == 0) {
        ctx->current_offset += seq.len;
        return true;
    }
    return false;
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
        .span.offset = ctx->current_offset,  
        .span.len = 1,
    };

    ctx->current_offset++;

    return tok;
}

lg_force_inline void
mrv_lexer_skip_whitespace(MRV_LexerContext *ctx) {
    for (
        uint8_t ch_i = ctx->text.p[ctx->current_offset];
        lg_char_is_whitespace(ch_i) && ctx->current_offset < ctx->text.len;
        ctx->current_offset++, ch_i = ctx->text.p[ctx->current_offset]
    );
}


lg_force_inline MRV_Token
mrv_lexer_scan_ident(MRV_LexerContext *ctx, MRV_TokenKind expected_kind) {
    MRV_Token tok = { 
        .kind = expected_kind,
        .span.offset = ctx->current_offset,
    };

    /////////////////////////////////////////////////////
    /// ~~ scan sequence ~~

    bool is_first = true;
    while (ctx->current_offset < ctx->text.len) {
        uint8_t ch_i = ctx->text.p[ctx->current_offset];

        if (lg_unlikely(ctx->current_offset >= ctx->text.len)) {
            mrv_report_error(&ctx->err, tok.span, lg_str8_lit("unexpected EOF"));
            return (MRV_Token){ .kind = MRV_TokenKind_Error };
        } 
        if (lg_unlikely(
            is_first &&
            expected_kind == MRV_TokenKind_Ident &&
            lg_char_is_numeric(ch_i)
        )) {
            mrv_report_error(&ctx->err, tok.span, lg_str8_lit("expected letter, found number"));
            return (MRV_Token){ .kind = MRV_TokenKind_Error };
        }
        if (lg_unlikely(!lg_char_is_alphanumeric(ch_i) && ch_i != '_')) {
            break;
        }

        is_first = false;
        ctx->current_offset++;
        tok.span.len++;

    }

    if (tok.span.len == 0) {
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

MRV_TokenStream
mrv_lex(LG_Allocator *artifact_allocator, lg_str8 text, LG_Writer *err_writer) {
    MRV_LexerContext ctx = {
        .artifact = artifact_allocator,
        .text = text,
        .err.writer = err_writer,
    };
    MRV_TokenStream tstream = {0};

    while (ctx.current_offset < ctx.text.len) {
        mrv_lexer_skip_whitespace(&ctx);

        switch (ctx.text.p[ctx.current_offset]) {
        case 'A'...'Z':
        case 'a'...'z': {
            MRV_Token sym_ident = mrv_lexer_scan_ident(&ctx, MRV_TokenKind_Ident);
            mrv_tstream_append(&tstream, ctx.artifact, sym_ident);

            break; 
        }

        case '%': {
            mrv_lexer_skip(&ctx);

            MRV_Token sym_ident = mrv_lexer_scan_ident(&ctx, MRV_TokenKind_SymbolIdent);
            mrv_tstream_append(&tstream, ctx.artifact, sym_ident);

            break;
        }

        case '-': {
            uint8_t next_ch = mrv_lexer_peek(&ctx);
            if (next_ch == '>') {
                MRV_Token tok = {
                    .kind = MRV_TokenKind_RightArrow,
                    .span.offset = ctx.current_offset,  
                    .span.len = 2,
                };
                ctx.current_offset += 2;
                mrv_tstream_append(&tstream, ctx.artifact, tok);
            } else {
                goto unexpected_char;
            }

            break;
        }

        case '(': {
            if (!mrv_lexer_match_sequence(&ctx, lg_str8_lit("(*"))) {
                goto single_char;
            }
            mrv_lexer_skip(&ctx);
            mrv_lexer_skip(&ctx);
            const lg_str8 close = lg_str8_lit("*)");
            while (!mrv_lexer_match_sequence(&ctx, close)) {
                mrv_lexer_skip(&ctx);
            }
            mrv_lexer_skip(&ctx);

            break;
        }

single_char:
        case ')':
        case '{':
        case '}':
        case ':':
        case ',':
        case ';':
        case '=': 
        case '<': 
        case '>': {
            MRV_Token ch = mrv_lexer_consume_char(&ctx);
            mrv_tstream_append(&tstream, ctx.artifact, ch);

            break;
        }

        case '\0':
            mrv_lexer_skip(&ctx);
            break;

unexpected_char:;
        default: {
            MRV_Span err_span = (MRV_Span){ .len = 1, .offset = ctx.current_offset };
            lg_str8 unexpected_char = mrv_span_to_str8(err_span, ctx.text);

            mrv_tstream_append(&tstream, ctx.artifact, (MRV_Token){
                .kind = MRV_TokenKind_Error,
                .span = err_span,
            });
            mrv_report_error(&ctx.err, err_span, lg_str8_lit("unexpected character: %{str}"), unexpected_char);

            mrv_lexer_skip(&ctx);
        }
        }
    }

    mrv_tstream_append(&tstream, ctx.artifact, (MRV_Token){
        .kind = MRV_TokenKind_EOF,
        .span = mrv_span_zero_len(ctx.current_offset),
    });

    return tstream;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// fundamental parsing data structures
///
////////////////////////////////////////////////////////////////////////////////

// the below parsing code is some of the grossest code on planet earth.
// change things with caution.

typedef uint8_t
MRV_ParserStatusKind;
enum {
    MRV_ParserStatusKind_OK,
    MRV_ParserStatusKind_NOK,
};

#define MRV_DEFINE_AST_NODE_KINDS \
    MRV_X(Error) \
    MRV_X(Program) \
    MRV_X(StencilIdent) \
    MRV_X(LanguageIdent) \
    MRV_X(SymbolIdent) \
    MRV_X(TypeIdent) \
    MRV_X(HostSymbolIdent) \
    MRV_X(HostTypeIdent) \
    MRV_X(OperatorIdent) \
    MRV_X(TypeName) \
    MRV_X(SymbolDeclaration) \
    MRV_X(LanguageDeclaration) \
    MRV_X(StencilDeclaration) \
    MRV_X(TypeDeclaration) \
    MRV_X(OperatorDeclaration) \
    MRV_X(OperatorDeclarationArg) \
    MRV_X(OperatorInvocation) \
    MRV_X(StencilArg) \
    MRV_X(StencilArgList) \
    MRV_X(Block) \
    MRV_X(Statement)

typedef uint8_t
MRV_ASTNodeKind;
enum 
MRV_ASTNodeKind {
#   define MRV_X(kind, ...) MRV_ASTNodeKind_##kind,
    MRV_DEFINE_AST_NODE_KINDS
#   undef MRV_X
};

enum {
#   define MRV_X(...) + 1
    MRV_ASTNodeKind_COUNT = 0 MRV_DEFINE_AST_NODE_KINDS
#   undef MRV_X
};

lg_str8
MRV_AST_NODE_STRINGS[MRV_ASTNodeKind_COUNT] = {
#   define MRV_X(kind, ...) [MRV_ASTNodeKind_##kind] = lg_str8_lit(#kind),
    MRV_DEFINE_AST_NODE_KINDS
#   undef MRV_X
};

#define mrv_ast_node_kind_as_str(kind) MRV_AST_NODE_STRINGS[(kind)]
#define mrv_match_ast_node(kind) switch((enum MRV_ASTNodeKind)kind)

typedef struct MRV_ASTNode MRV_ASTNode;

typedef union 
MRV_ASTNodeChildren {
    struct {} Error;

    struct {
        size_t         n_children;
        MRV_ASTNode  **children;
    } Program;

    struct {} StencilIdent;
    struct {} LanguageIdent;
    struct {} SymbolIdent;
    struct {} TypeIdent;
    struct {} HostSymbolIdent;
    struct {} HostTypeIdent;
    struct {} OperatorIdent;

    struct {
        MRV_ASTNode *outer_ident;
        MRV_ASTNode *inner_typename;
    } TypeName;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *left_arg;
        MRV_ASTNode *right_arg;
    } OperatorInvocation;

    struct {
        MRV_ASTNode *symbol_ident;
        MRV_ASTNode *type_ident;
    } SymbolDeclaration;

    struct {
        MRV_ASTNode *ident;
    } LanguageDeclaration;

    struct {
        MRV_ASTNode *ident;
    } TypeDeclaration;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *arg_list;
        MRV_ASTNode *left_arg;
        MRV_ASTNode *right_arg;
        MRV_ASTNode *return_typename;
    } OperatorDeclaration;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *type;
    } OperatorDeclarationArg;
    
    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *arg_list;
        MRV_ASTNode *body;
    } StencilDeclaration;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *host_type;
    } StencilArg;

    struct {
        size_t         n_args;
        MRV_ASTNode  **args;
    } StencilArgList;

    struct {
        size_t         n_statements;
        MRV_ASTNode  **statements;
    } Block;

    struct {
        MRV_ASTNode *symbol_decl;
        MRV_ASTNode *operator_invocation;
    } Statement;
} MRV_ASTNodeChildren;

struct 
MRV_ASTNode {
    MRV_ASTNodeChildren children_as;

    MRV_ASTNodeKind kind;
    MRV_Span span;
};

typedef struct
MRV_ASTNodeRefStack {
    struct MRV_ASTNodeRefStack *prev;
    MRV_ASTNode *to;
} MRV_ASTNodeRefStack;

typedef struct
MRV_ParserContext {
    lg_str8                text;

    MRV_ASTNodeRefStack   *ref_stack_top;

    MRV_TokenStream       *tstream;
    MRV_TokenStreamBlock  *cur_block;
    MRV_Token             *cur_token;

    MRV_Error              err;
    LG_Arena               scratch;

    LG_Arena               artifact;
    MRV_ASTNode           *nil_node;
} MRV_ParserContext;

typedef struct
MRV_AST {
    MRV_ASTNode  *nil_node;
    MRV_ASTNode  *root;
    LG_Arena      artifact;
} MRV_AST;

#define mrv_parser_has_err(ctx) ((ctx)->err.is_err)


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// iterate over a token stream
///
////////////////////////////////////////////////////////////////////////////////

lg_force_inline MRV_ParserStatusKind 
mrv_parser_find_next(
    MRV_ParserContext *ctx,
    MRV_Token **out_next_token,
    MRV_TokenStreamBlock **out_next_block
) {
    lg_assert(ctx != NULL);
    lg_assert(ctx->cur_block != NULL);
    lg_assert(out_next_block != NULL);
    lg_assert(out_next_token != NULL);

    if (ctx->cur_token == NULL) {
        *out_next_block = ctx->cur_block;
        *out_next_token = ctx->cur_block->tokens;
        return MRV_ParserStatusKind_OK;
    }

    const ptrdiff_t cur_offset = ctx->cur_token - ctx->cur_block->tokens;
    lg_assert(cur_offset >= 0);

    const size_t cur_len = ctx->cur_block == ctx->tstream->tail ? 
        ctx->tstream->tail_len :
        MRV_TOKEN_STREAM_BLOCK_CAPACITY;

    if (lg_likely((size_t)cur_offset < cur_len)) {
        *out_next_block = ctx->cur_block;
        *out_next_token = ctx->cur_token + 1;
        return MRV_ParserStatusKind_OK;
    }

    if (ctx->cur_block->next == NULL) {
        return MRV_ParserStatusKind_NOK;
    }

    *out_next_block = ctx->cur_block->next;
    *out_next_token = ctx->cur_block->next->tokens;

    return true;
}

MRV_Token
mrv_parser_consume(MRV_ParserContext *ctx) {
    lg_assert(ctx != NULL);

    MRV_Token *next_token;
    MRV_TokenStreamBlock *next_block;
    MRV_ParserStatusKind status = mrv_parser_find_next(ctx, &next_token, &next_block);
    lg_assert(status == MRV_ParserStatusKind_OK); // this function shouldn't be called where there isn't a next token
                         // e.g don't expect a token after an EOF
                         // TODO: this assumption may case nodes with variadic children left unterminated
                         // to crash the parser

    ctx->cur_token = next_token;
    ctx->cur_block = next_block;

    return *next_token;
}

MRV_Token
mrv_parser_expect(
    MRV_ParserContext *ctx,
    MRV_TokenKind expected_kind
) {
    lg_assert(ctx != NULL);
    lg_assert(expected_kind != MRV_TokenKind_Error);

    MRV_Token next_token = mrv_parser_consume(ctx);

    if (expected_kind != next_token.kind) {
        mrv_report_error(
            &ctx->err,
            next_token.span,
            lg_str8_lit("expected %{str}, found string \"%{str}\" of token kind %{str}"), 
            mrv_token_as_str(expected_kind),    
            mrv_span_to_str8(next_token.span, ctx->text),
            mrv_token_as_str(next_token.kind)
        );
        return lg_nil(MRV_Token);        
    }

    return next_token;
}

MRV_Token
mrv_parser_peek(MRV_ParserContext *ctx) {
    MRV_Token *next_token;
    MRV_TokenStreamBlock *next_block;
    MRV_ParserStatusKind status = mrv_parser_find_next(ctx, &next_token, &next_block);
    if (status != MRV_ParserStatusKind_OK) {
        return lg_nil(MRV_Token);
    }

    return *next_token;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// rdp helpers
///
////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// note: with these nrs helpers, you still have to use lg_push/pop_scope

lg_force_inline MRV_ASTNode*
mrv_parser_nil_node(MRV_ParserContext *ctx) {
    lg_assert(ctx != NULL);
    lg_assert(ctx->nil_node != NULL);
    lg_memzero(ctx->nil_node, sizeof(MRV_ASTNode));
    return ctx->nil_node;
}

lg_force_inline bool
mrv_parser_is_nil_node(MRV_ParserContext *ctx, MRV_ASTNode *node) {
    lg_assert(ctx != NULL);
    lg_assert(ctx->nil_node != NULL);
    return node == ctx->nil_node;
}

lg_force_inline void
mrv_parser_nrs_push(MRV_ParserContext *ctx, MRV_ASTNode *to) {
    MRV_ASTNodeRefStack* to_push = (MRV_ASTNodeRefStack*)lg_arena_alloc_struct(&ctx->scratch, MRV_ASTNodeRefStack);
    lg_assert(to_push != NULL);

    to_push->to = to;

    if (ctx->ref_stack_top != NULL) {
        to_push->prev = ctx->ref_stack_top;
    }
    ctx->ref_stack_top = to_push;
}

lg_force_inline MRV_ASTNode**
mrv_parser_nrs_unwind_cpy(MRV_ParserContext *ctx, uint32_t n_refs) {
    lg_assert(n_refs != 0); // doesn't handle this well yet

    MRV_ASTNode **refs = lg_arena_alloc_array(&ctx->artifact, MRV_ASTNode*, n_refs);
    lg_assert(refs != NULL);

    uint32_t i = 0;
    while (ctx->ref_stack_top != NULL && i < n_refs) {
        MRV_ASTNode *next = ctx->ref_stack_top->to;
        refs[i] = next;

        ctx->ref_stack_top = ctx->ref_stack_top->prev;
        i++;
    }

    return refs;
}

#define mrv_parser_mknode(ctx, kind, span, ...) mrv_parser_mknode_( \
    ctx, \
    MRV_ASTNodeKind_##kind, \
    span, \
    (MRV_ASTNodeChildren){ .kind = {__VA_ARGS__} } \
)

lg_force_inline MRV_ASTNode*
mrv_parser_mknode_(MRV_ParserContext *ctx, MRV_ASTNodeKind kind, MRV_Span span, MRV_ASTNodeChildren children) {
    MRV_ASTNode *node = lg_arena_alloc_struct(&ctx->artifact, MRV_ASTNode);
    lg_assert(node != NULL);

    node->kind = kind;
    node->span = span;
    node->children_as = children;

    return node;
}

lg_force_inline MRV_Span
mrv_get_bounding_span(size_t n_nodes, MRV_ASTNode **nodes) {
    lg_assert(n_nodes != 0);

    size_t min_offset = SIZE_MAX,
           max_offset = 0;
    for (size_t i = 0; i < n_nodes; i++) {
        if (nodes[i]->span.offset < min_offset) {
            min_offset = nodes[i]->span.offset;
        }

        size_t offset = nodes[i]->span.offset + nodes[i]->span.len;
        if (offset > max_offset) {
            max_offset = offset;
        }
    }

    lg_assert(min_offset != SIZE_MAX);
    lg_assert(max_offset != 0);
    lg_assert(min_offset < max_offset);

    return (MRV_Span){ .offset = min_offset, .len = max_offset - min_offset };
}

lg_force_inline void
mrv_parser_unexpected_token(MRV_ParserContext *ctx, MRV_Token tok) {
    mrv_report_error(
        &ctx->err,
        tok.span,
        lg_str8_lit("unexpected token: \"%{str}\" of token kind %{str}"),
        mrv_span_to_str8(tok.span, ctx->text),
        mrv_token_as_str(tok.kind)
    );
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// actual rdp procedures
///
////////////////////////////////////////////////////////////////////////////////

MRV_ASTNode*
mrv_parse_stencil_ident(MRV_ParserContext *ctx) {
    MRV_Token name = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, StencilIdent, name.span);
    return node;
}

MRV_ASTNode*
mrv_parse_host_symbol_ident(MRV_ParserContext *ctx) {
    MRV_Token ident = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, HostSymbolIdent, ident.span);
    return node;
}

MRV_ASTNode*
mrv_parse_host_type_ident(MRV_ParserContext *ctx) {
    MRV_Token ident = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, HostTypeIdent, ident.span);
    return node;
}


MRV_ASTNode*
mrv_parse_symbol_ident(MRV_ParserContext *ctx) {
    MRV_Token tok = mrv_parser_expect(ctx, MRV_TokenKind_SymbolIdent);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, SymbolIdent, tok.span);
    return node;
}

MRV_ASTNode* 
mrv_parse_type_ident(MRV_ParserContext *ctx) {
    MRV_Token tok = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, TypeIdent, tok.span);
    return node;
}

MRV_ASTNode*
mrv_parse_type_name(MRV_ParserContext *ctx, bool parent_is_host) {
    // our job here isn't to validate that host types require a parameter,
    // but just to make sure that type identifiers inside Host<...> are parsed 
    // as HostTypeIdent
    if (parent_is_host) {
        MRV_ASTNode *host_type = mrv_parse_host_type_ident(ctx);
        MRV_ASTNode *node = mrv_parser_mknode(
            ctx,
            TypeName,
            host_type->span,
            .outer_ident = host_type,
            .inner_typename = mrv_parser_nil_node(ctx),
        );
        return node;
    }

    MRV_ASTNode *outer_ident = mrv_parse_type_ident(ctx);
    lg_str8 node_text = mrv_span_to_str8(outer_ident->span, ctx->text);
    bool is_host = lg_strcmp(node_text, lg_str8_lit("Host"));

    MRV_Token peek = mrv_parser_peek(ctx);

    if (peek.kind != MRV_TokenKind_OpenAngleBracket) {
        MRV_ASTNode *node = mrv_parser_mknode(
            ctx,
            TypeName,
            outer_ident->span,
            .outer_ident = outer_ident,
            .inner_typename = mrv_parser_nil_node(ctx),
        );
        return node;
    }

    mrv_parser_consume(ctx); // '<'
    MRV_ASTNode *inner_typename = mrv_parse_type_name(ctx, is_host);
    mrv_parser_expect(ctx, MRV_TokenKind_CloseAngleBracket);

    MRV_Span all_span = mrv_get_bounding_span(2, (MRV_ASTNode*[]){outer_ident, inner_typename});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        TypeName,
        all_span,
        .inner_typename = inner_typename,
        .outer_ident = outer_ident,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_symbol_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *symbol_ident = mrv_parse_symbol_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Colon);
    MRV_ASTNode *type_name = mrv_parse_type_name(ctx, false);

    MRV_Span all_span = mrv_get_bounding_span(2, (MRV_ASTNode*[]){symbol_ident, type_name});
    MRV_ASTNode *node = mrv_parser_mknode(ctx, SymbolDeclaration, all_span, symbol_ident, type_name);

    return node;
}

MRV_ASTNode* 
mrv_parse_operator_ident(MRV_ParserContext *ctx) {
    MRV_Token tok = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, OperatorIdent, tok.span);
    return node;
}

// forward decl b/c blocks can contain blocks 
MRV_ASTNode*
mrv_parse_block(MRV_ParserContext *ctx);

MRV_ASTNode*
mrv_parse_operator_invocation(MRV_ParserContext *ctx) {
    MRV_ASTNode *op_ident = mrv_parse_operator_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);
    MRV_ASTNode *left_arg = mrv_parser_nil_node(ctx);
    MRV_ASTNode *right_arg = mrv_parser_nil_node(ctx);

    // currently, there can only be two args in an op invocation
    while (true) {
        MRV_Token peek = mrv_parser_peek(ctx);
        switch (peek.kind) {
        case MRV_TokenKind_SymbolIdent: {
            if (mrv_parser_is_nil_node(ctx, left_arg)) {
                left_arg = mrv_parse_symbol_ident(ctx);
            } else {
                lg_assert(mrv_parser_is_nil_node(ctx, right_arg));
                right_arg = mrv_parse_symbol_ident(ctx);
            }
            break;
        }

        case MRV_TokenKind_Ident: {
            if (mrv_parser_is_nil_node(ctx, left_arg)) {
                left_arg = mrv_parse_host_symbol_ident(ctx);
            } else {
                lg_assert(mrv_parser_is_nil_node(ctx, right_arg));
                right_arg = mrv_parse_host_symbol_ident(ctx);
            }
            break;
        }

        case MRV_TokenKind_OpenBrace: {
            mrv_parser_consume(ctx);
            if (mrv_parser_is_nil_node(ctx, left_arg)) {
                left_arg = mrv_parse_block(ctx);
            } else {
                lg_assert(mrv_parser_is_nil_node(ctx, right_arg));
                right_arg = mrv_parse_block(ctx);
            }
            break;
        }

        // TODO: currently, this means you can just write
        // MyOperator(,,,,,,,,)
        // this may or may not actually be a problem
        case MRV_TokenKind_Comma:
            mrv_parser_consume(ctx);
            break;

        case MRV_TokenKind_CloseParen:
            mrv_parser_consume(ctx);
            goto loop_end;
            
        default: {
            mrv_parser_unexpected_token(ctx, peek);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
        }
    }
loop_end:;

    MRV_Span all_span = mrv_get_bounding_span(3, (MRV_ASTNode*[]){op_ident, left_arg, right_arg});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        OperatorInvocation,
        all_span,
        .ident = op_ident,
        .left_arg = left_arg,
        .right_arg = right_arg,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_statement(MRV_ParserContext *ctx) {
    MRV_ASTNode *symbol_decl = mrv_parser_nil_node(ctx);

    MRV_Token peek = mrv_parser_peek(ctx);
    switch (peek.kind) {
    case MRV_TokenKind_SymbolIdent: {
        symbol_decl = mrv_parse_symbol_decl(ctx);
        mrv_parser_expect(ctx, MRV_TokenKind_Equals);
        break;
    }

    case MRV_TokenKind_Ident: {
        break;
    };

    default: {
        mrv_parser_unexpected_token(ctx, peek);
        mrv_parser_consume(ctx);
    }
    }

    MRV_ASTNode *operator_invocation = mrv_parse_operator_invocation(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_ASTNode *first_node = mrv_parser_is_nil_node(ctx, symbol_decl) ?
        operator_invocation :
        symbol_decl;
    MRV_Span all_span = mrv_get_bounding_span(2, (MRV_ASTNode*[]){first_node, operator_invocation});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        Statement,
        all_span,
        .symbol_decl = symbol_decl,
        .operator_invocation = operator_invocation
    );

    return node; 
}

MRV_ASTNode*
mrv_parse_block(MRV_ParserContext *ctx) {
    LG_Scope scope = lg_push_scope(&ctx->scratch);

    uint32_t n_children = 0;

    while (true) {
        MRV_Token peek = mrv_parser_peek(ctx);
        switch (peek.kind) {
        case MRV_TokenKind_Ident:
        case MRV_TokenKind_SymbolIdent: {
            MRV_ASTNode *stmt = mrv_parse_statement(ctx);
            mrv_parser_nrs_push(ctx, stmt);
            n_children++;
            break;
        }

        case MRV_TokenKind_CloseBrace: {
            mrv_parser_consume(ctx);
            goto loop_end;
        }

        default:
            mrv_parser_unexpected_token(ctx, peek);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(n_children, children);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, Block, all_span, .n_statements = n_children, .statements = children);

    lg_pop_scope(&ctx->scratch, scope);
    return node;
}

MRV_ASTNode*
mrv_parse_stencil_arg(MRV_ParserContext *ctx) {
    MRV_ASTNode* name = mrv_parse_host_symbol_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Colon);
    MRV_ASTNode* type = mrv_parse_type_name(ctx, false);
    
    MRV_Span all_span = mrv_get_bounding_span(2, (MRV_ASTNode*[]){name ,type});
    MRV_ASTNode *node = mrv_parser_mknode(ctx, StencilArg, all_span, .ident = name, .host_type = type);

    return node;
}

MRV_ASTNode*
mrv_parse_stencil_arg_list(MRV_ParserContext *ctx) {
    LG_Scope scope = lg_push_scope(&ctx->scratch);

    uint32_t n_children = 0;

    while (true) {
        MRV_Token peek = mrv_parser_peek(ctx);
        switch (peek.kind) {
        case MRV_TokenKind_CloseParen:
            mrv_parser_consume(ctx);
            goto loop_end;
        case MRV_TokenKind_Comma:
            mrv_parser_consume(ctx);
            break;
        case MRV_TokenKind_Ident: {
            MRV_ASTNode *arg = mrv_parse_stencil_arg(ctx);
            mrv_parser_nrs_push(ctx, arg);
            n_children++;
            break;
        }
        default:
            mrv_parser_unexpected_token(ctx, peek);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(n_children, children);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, StencilArgList, all_span, .args = children, .n_args = n_children);

    lg_pop_scope(&ctx->scratch, scope);

    return node;
}

MRV_ASTNode*
mrv_parse_stencil_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_stencil_ident(ctx);

    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);
    MRV_ASTNode *arg_list = mrv_parse_stencil_arg_list(ctx);

    mrv_parser_expect(ctx, MRV_TokenKind_OpenBrace);
    MRV_ASTNode *body = mrv_parse_block(ctx);

    if (mrv_parser_has_err(ctx)) {
        return mrv_parser_nil_node(ctx);
    }

    MRV_Span all_span = mrv_get_bounding_span(3, (MRV_ASTNode*[]){ident, arg_list, body});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        StencilDeclaration,
        all_span,
        .ident = ident,
        .arg_list = arg_list,
        .body = body,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_language_ident(MRV_ParserContext *ctx) {
    MRV_Token tok = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, LanguageIdent, tok.span);
    return node;
}

MRV_ASTNode*
mrv_parse_language_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_language_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, LanguageDeclaration, ident->span, .ident = ident);
    return node;
}

MRV_ASTNode*
mrv_parse_type_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_type_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, TypeDeclaration, ident->span, .ident = ident);
    return node;
}

MRV_ASTNode*
mrv_parse_operator_decl_arg(MRV_ParserContext *ctx) {
    MRV_ASTNode *name = mrv_parse_host_symbol_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Colon);
    MRV_ASTNode *type = mrv_parse_type_name(ctx, false);
    
    MRV_Span all_span = mrv_get_bounding_span(2, (MRV_ASTNode*[]){name ,type});
    MRV_ASTNode *node = mrv_parser_mknode(ctx, OperatorDeclarationArg, all_span, .ident = name, .type = type);

    return node;
}

MRV_ASTNode*
mrv_parse_operator_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_operator_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);

    size_t n_args = 0;
    MRV_ASTNode *args[2] = {mrv_parser_nil_node(ctx), mrv_parser_nil_node(ctx)};

    while (true) {
        MRV_Token peek = mrv_parser_peek(ctx);
        switch (peek.kind) {
        case MRV_TokenKind_CloseParen:
            mrv_parser_consume(ctx);
            goto end_loop;
        
        case MRV_TokenKind_Ident: {
            args[n_args] = mrv_parse_operator_decl_arg(ctx);
            n_args++;
            break;
        }

        case MRV_TokenKind_Comma: {
            mrv_parser_consume(ctx);
            break;
        }

        default: {
            mrv_parser_unexpected_token(ctx, peek);
            mrv_parser_consume(ctx);
            MRV_ASTNode *node = mrv_parser_mknode(ctx, Error, peek.span,);
            return node;
        }
        }

        if (n_args >= 2) {
            mrv_parser_expect(ctx, MRV_TokenKind_CloseParen);
            break;
        }
    }
end_loop:;

    MRV_ASTNode *typename = mrv_parser_nil_node(ctx);
    MRV_Token peek = mrv_parser_peek(ctx);
    if (peek.kind == MRV_TokenKind_RightArrow) {
        mrv_parser_consume(ctx);
        typename = mrv_parse_type_name(ctx, false);
    }

    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        OperatorDeclaration,
        ident->span,
        .ident = ident,
        .left_arg = args[0],
        .right_arg = args[1],
        .return_typename = typename,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_program(MRV_ParserContext *ctx) {
    LG_Scope scope = lg_push_scope(&ctx->scratch);
    MRV_ASTNode *root = mrv_parser_nil_node(ctx);
    size_t n_children = 0;

    MRV_Token tok = mrv_parser_peek(ctx);
    while (true) {
        switch (tok.kind) {
        case MRV_TokenKind_EOF: {
            goto loop_end;
        }

        case MRV_TokenKind_Stencil: {
            mrv_parser_consume(ctx);
            MRV_ASTNode *stencil = mrv_parse_stencil_decl(ctx);
            mrv_parser_nrs_push(ctx, stencil);
            break;
        }

        case MRV_TokenKind_Language: {
            mrv_parser_consume(ctx);
            MRV_ASTNode *language_decl = mrv_parse_language_decl(ctx);
            mrv_parser_nrs_push(ctx, language_decl);
            break;
        }

        case MRV_TokenKind_Type: {
            mrv_parser_consume(ctx);
            MRV_ASTNode *type_decl = mrv_parse_type_decl(ctx);
            mrv_parser_nrs_push(ctx, type_decl);
            break;
        }

        case MRV_TokenKind_Operator: {
            mrv_parser_consume(ctx);
            MRV_ASTNode *operator_decl = mrv_parse_operator_decl(ctx);
            mrv_parser_nrs_push(ctx, operator_decl);
            break;
        }

        case MRV_TokenKind_Error: 
            lg_unreachable();

        default: {
            mrv_parser_unexpected_token(ctx, tok);
            mrv_parser_consume(ctx);
            root = mrv_parser_mknode(ctx, Error, tok.span);
            goto out;
        }
        }

        n_children++;
        tok = mrv_parser_peek(ctx);
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(n_children, children);
    root = mrv_parser_mknode(ctx, Program, all_span, .n_children = n_children, children = children);
    lg_assert(root != NULL);

    root->kind = MRV_ASTNodeKind_Program;
    root->span = tok.span;

out:
    lg_pop_scope(&ctx->scratch, scope);
    return root;
}

MRV_AST
mrv_parse(
    LG_Allocator *artifact_allocator,
    LG_Allocator *scratch_allocator, 
    LG_Writer *err_writer,
    MRV_TokenStream *tstream,
    lg_str8 text
) {
    lg_assert(artifact_allocator != NULL);
    lg_assert(scratch_allocator != NULL);
    lg_assert(tstream != NULL);
    lg_assert(tstream->tail->next == NULL);

    
    /////////////////////////////////////
    /// ~~ initialize the parser ~~

    MRV_ParserContext ctx = {
        .tstream = tstream,
        .err.writer = err_writer,
        .text = text,
    };

    // this will loop forever if there are cycles
    MRV_TokenStreamBlock *iter_block = tstream->tail;
    while (iter_block != NULL) {
        ctx.cur_block = iter_block;
        iter_block = iter_block->prev;
    }

    lg_arena_init(&ctx.artifact, artifact_allocator);
    lg_arena_init(&ctx.scratch, scratch_allocator);

    ctx.nil_node = lg_arena_alloc_struct(&ctx.artifact, MRV_ASTNode);
    lg_assert(ctx.nil_node != NULL);


    /////////////////////////////////////
    /// ~~ do the parsing ~~

    MRV_ASTNode *root = mrv_parse_program(&ctx);
    MRV_AST ast = {
        .root = root,
        .nil_node = ctx.nil_node,
        .artifact = ctx.artifact,
    };

    lg_arena_free_all(&ctx.scratch);

    return ast;
}

void
mrv_ast_destroy(MRV_AST *ast) {
    lg_arena_free_all(&ast->artifact);
    lg_memzero(ast, sizeof(MRV_AST));
}

lg_force_inline bool
mrv_ast_is_root(MRV_AST *ast, MRV_ASTNode *node) {
    lg_assert(ast != NULL);
    lg_assert(ast->root != NULL);
    return node == ast->root;
}

lg_force_inline bool
mrv_ast_is_nil_node(MRV_AST *ast, MRV_ASTNode *node) {
    lg_assert(ast != NULL);
    lg_assert(ast->nil_node != NULL);
    return node == ast->nil_node;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// dump the ast
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_ASTDumpContext {
    MRV_AST    *ast;
    LG_Writer  *writer;
    lg_str8     text;
    int8_t      indent;
} MRV_ASTDumpContext;

lg_force_inline void
mrv_ast_dump_indent(MRV_ASTDumpContext *ctx) {
    for (int8_t i = 0; i < ctx->indent; i++) {
        lg_write(ctx->writer, lg_str8_lit("    "));
    }
}

void
mrv_ast_dump_r(MRV_ASTDumpContext *ctx, MRV_ASTNode *lg_nullable parent, MRV_ASTNode *self) {
    if (mrv_ast_is_nil_node(ctx->ast, self)) {
        return;
    }

    bool is_leaf = false;
    MRV_ASTNodeChildren as = self->children_as;

    mrv_match_ast_node (self->kind) {
        case MRV_ASTNodeKind_Error:
        case MRV_ASTNodeKind_StencilIdent:
        case MRV_ASTNodeKind_LanguageIdent:
        case MRV_ASTNodeKind_SymbolIdent:
        case MRV_ASTNodeKind_TypeIdent:
        case MRV_ASTNodeKind_HostSymbolIdent:
        case MRV_ASTNodeKind_HostTypeIdent:
        case MRV_ASTNodeKind_OperatorIdent:
            is_leaf = true;
            break;

        case MRV_ASTNodeKind_LanguageDeclaration:
            mrv_ast_dump_r(ctx, self, as.LanguageDeclaration.ident);
            break;

        case MRV_ASTNodeKind_TypeDeclaration:
            mrv_ast_dump_r(ctx, self, as.TypeDeclaration.ident);
            break;

        case MRV_ASTNodeKind_OperatorDeclaration:
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.ident);
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.left_arg);
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.right_arg);
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.return_typename);
            break;

        case MRV_ASTNodeKind_OperatorDeclarationArg:
            mrv_ast_dump_r(ctx, self, as.OperatorDeclarationArg.ident);
            mrv_ast_dump_r(ctx, self, as.OperatorDeclarationArg.type);
            break;

        case MRV_ASTNodeKind_Program:
            for (uint32_t i = 0; i < as.Program.n_children; i++) {
                mrv_ast_dump_r(ctx, self, as.Program.children[i]);
            }
            break;

        case MRV_ASTNodeKind_SymbolDeclaration:
            mrv_ast_dump_r(ctx, self, as.SymbolDeclaration.symbol_ident);
            mrv_ast_dump_r(ctx, self, as.SymbolDeclaration.type_ident);
            break;

        case MRV_ASTNodeKind_OperatorInvocation:
            mrv_ast_dump_r(ctx, self, as.OperatorInvocation.ident);
            mrv_ast_dump_r(ctx, self, as.OperatorInvocation.left_arg);
            mrv_ast_dump_r(ctx, self, as.OperatorInvocation.right_arg);
            break;

        case MRV_ASTNodeKind_StencilDeclaration:
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.ident);
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.arg_list);
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.body);
            break;

        case MRV_ASTNodeKind_StencilArg:
            mrv_ast_dump_r(ctx, self, as.StencilArg.ident);
            mrv_ast_dump_r(ctx, self, as.StencilArg.host_type);
            break;

        case MRV_ASTNodeKind_StencilArgList:
            for (uint32_t i = 0; i < as.StencilArgList.n_args; i++) {
                mrv_ast_dump_r(ctx, self, as.StencilArgList.args[i]);
            }
            break;

        case MRV_ASTNodeKind_Block:
            lg_write(ctx->writer, lg_str8_lit("\n"));
            mrv_ast_dump_indent(ctx);
            lg_printf(ctx->writer, lg_str8_lit("subgraph \"cluster_%{i64}\" {\n"), self);
            ctx->indent++;
            for (uint32_t i = 0; i < as.Block.n_statements; i++) {
                mrv_ast_dump_r(ctx, self, as.Block.statements[i]);
            }
            ctx->indent--;
            mrv_ast_dump_indent(ctx);
            lg_write(ctx->writer, lg_str8_lit("}\n\n"));
            break;

        case MRV_ASTNodeKind_Statement:
            mrv_ast_dump_r(ctx, self, as.Statement.symbol_decl);
            mrv_ast_dump_r(ctx, self, as.Statement.operator_invocation);
            break;

        case MRV_ASTNodeKind_TypeName:
            mrv_ast_dump_r(ctx, self, as.TypeName.outer_ident);
            mrv_ast_dump_r(ctx, self, as.TypeName.inner_typename);
            break;
        }

    mrv_ast_dump_indent(ctx);
    lg_printf(ctx->writer, lg_str8_lit("\"node_%{i64}\""), self);
    lg_str8 kind_str = mrv_ast_node_kind_as_str(self->kind);
    if (is_leaf) {
        lg_printf(ctx->writer, lg_str8_lit(" [label=\"%{str} (\\\"%{str}\\\")\"];\n"), kind_str, mrv_span_to_str8(self->span, ctx->text));
    } else {
        lg_printf(ctx->writer, lg_str8_lit(" [label=\"%{str}\"];\n"), kind_str);
    }

    if (parent != NULL) {
        mrv_ast_dump_indent(ctx);
        lg_printf(ctx->writer, lg_str8_lit("\"node_%{i64}\""), parent);
        lg_write(ctx->writer, lg_str8_lit(" -> "));
        lg_printf(ctx->writer, lg_str8_lit("\"node_%{i64}\""), self);
        lg_write(ctx->writer, lg_str8_lit(";\n"));
    }
}

void
mrv_ast_dump(MRV_AST *ast, LG_Writer *writer, lg_str8 text) {
    MRV_ASTDumpContext ctx = {
        .ast = ast,
        .writer = writer,
        .text = text,
    };

    lg_write(writer, lg_str8_lit(
        "digraph Abstract_Syntax_Tree {\n"
        // "     size=\"11,8.5!\";\n"
        // "     ratio=\"fill\";\n"
        // "     rankdir=TB;\n\n"
    ));
    ctx.indent++;
    mrv_ast_dump_r(&ctx, NULL, ast->root);
    ctx.indent--;
    lg_write(writer, lg_str8_lit("}\n"));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// semantic analysis stuff
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_Symtab {
    LG_Table  table;
    lg_str8  *types;
} MRV_Symtab;

typedef struct
MRV_SemaContext {
    MRV_AST    *ast;
    lg_str8     text;
    MRV_Symtab  symtab;
    MRV_Error   err;
} MRV_SemaContext;

void
mrv_sema_typecheck_r(MRV_SemaContext *ctx, MRV_ASTNode *self) {
    if (mrv_ast_is_nil_node(ctx->ast, self)) {
        return;
    }

    MRV_ASTNodeChildren as = self->children_as;

    mrv_match_ast_node (self->kind) {
        case MRV_ASTNodeKind_OperatorDeclarationArg:
        case MRV_ASTNodeKind_Error:
        case MRV_ASTNodeKind_StencilIdent:
        case MRV_ASTNodeKind_SymbolIdent:
        case MRV_ASTNodeKind_TypeIdent:
        case MRV_ASTNodeKind_TypeDeclaration:
        case MRV_ASTNodeKind_OperatorDeclaration:
        case MRV_ASTNodeKind_HostSymbolIdent:
        case MRV_ASTNodeKind_HostTypeIdent:
        case MRV_ASTNodeKind_OperatorIdent:
        case MRV_ASTNodeKind_LanguageIdent:
        case MRV_ASTNodeKind_LanguageDeclaration:
            break;

        case MRV_ASTNodeKind_Program:
            for (uint32_t i = 0; i < as.Program.n_children; i++) {
                mrv_sema_typecheck_r(ctx, as.Program.children[i]);
            }
            break;

        case MRV_ASTNodeKind_SymbolDeclaration: {
            lg_str8 sym_ident = mrv_span_to_str8(as.SymbolDeclaration.symbol_ident->span, ctx->text);
            lg_str8 type_ident = mrv_span_to_str8(as.SymbolDeclaration.type_ident->span, ctx->text);

            size_t idx;
            bool found;
            LG_StatusKind status = lg_table_ensure_str8(&ctx->symtab.table, sym_ident, &idx, &found);
            lg_assert(status = LG_StatusKind_OK);

            if (found) {
                mrv_report_error(
                    &ctx->err,
                    self->span,
                    lg_str8_lit("multiple declarations found for symbol %{str}"),
                    sym_ident
                );
            } else {
                ctx->symtab.types[idx] = type_ident;
            }

            break;
        }

        case MRV_ASTNodeKind_OperatorInvocation:
            mrv_sema_typecheck_r(ctx, as.OperatorInvocation.ident);
            mrv_sema_typecheck_r(ctx, as.OperatorInvocation.left_arg);
            mrv_sema_typecheck_r(ctx, as.OperatorInvocation.right_arg);
            break;

        case MRV_ASTNodeKind_StencilDeclaration:
            mrv_sema_typecheck_r(ctx, as.StencilDeclaration.ident);
            mrv_sema_typecheck_r(ctx, as.StencilDeclaration.arg_list);
            mrv_sema_typecheck_r(ctx, as.StencilDeclaration.body);
            break;

        case MRV_ASTNodeKind_StencilArg:
            mrv_sema_typecheck_r(ctx, as.StencilArg.ident);
            mrv_sema_typecheck_r(ctx, as.StencilArg.host_type);
            break;

        case MRV_ASTNodeKind_StencilArgList:
            for (uint32_t i = 0; i < as.StencilArgList.n_args; i++) {
                mrv_sema_typecheck_r(ctx, as.StencilArgList.args[i]);
            }
            break;

        case MRV_ASTNodeKind_Block:
            for (uint32_t i = 0; i < as.Block.n_statements; i++) {
                mrv_sema_typecheck_r(ctx, as.Block.statements[i]);
            }
            break;

        case MRV_ASTNodeKind_Statement:
            mrv_sema_typecheck_r(ctx, as.Statement.symbol_decl);
            mrv_sema_typecheck_r(ctx, as.Statement.operator_invocation);
            break;
          break;
        case MRV_ASTNodeKind_TypeName:
          break;
        }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// actual entry point
///
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>

void*
alloc_libc(void *_, size_t bytes) {
    (void)_;
    return calloc(bytes, 1);
}

void 
free_libc(void* _, void *ptr) {
    (void)_;
    return free(ptr);
}

size_t
write_stdout(void *ctx, lg_str8 msg) {
    (void)ctx;
    return printf("%.*s", (int32_t)msg.len, msg.p);
}

static LG_Allocator 
libc_allocator = {
    .alloc = alloc_libc,
    .free = free_libc,
    .default_slab_size_bytes = 1024 * 1024 * 1024,
};

static LG_Writer 
libc_writer = {
    .write = write_stdout,
};

int main() {
    FILE *file = fopen("./tools/ir_test.mirv", "r+");
    lg_assert(file != NULL);

    uint8_t file_contents[4096] = {0};
    size_t chunks_read = fread(file_contents, sizeof(file_contents) / 4, 4, file);
    lg_assert(chunks_read > 0);

    lg_str8 text = (lg_str8){ .len = 4096, .p = file_contents };
    MRV_TokenStream tstream = mrv_lex(&libc_allocator, text, &libc_writer);

    (void)text;

    MRV_AST ast = mrv_parse(
        &libc_allocator,
        &libc_allocator,
        &libc_writer,
        &tstream,
        text
    );

    mrv_ast_dump(&ast, &libc_writer, text);

    mrv_tstream_destroy(&tstream, &libc_allocator);
    mrv_ast_destroy(&ast);
    fclose(file);
    return 0;
}

#include <libgrad/internal/base.c>
