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
    MRV_X(BeginHostType,      "<<") \
    MRV_X(EndHostType,        ">>") \
    MRV_X(Language,           "language") \
    MRV_X(Type,               "type") \
    MRV_X(Operator,           "operator") \
    MRV_X(Stencil,            "stencil") \
    MRV_X(ControlFlow,        "control_flow") \
    MRV_X(RightArrow,         "->") \
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
mrv_lexer_match_sequence(MRV_LexerContext *ctx, lg_str8 seq, MRV_Span *lg_nullable out_span) {
    if (ctx->current_offset + seq.len >= ctx->text.len) {
        return false;
    }
    lg_str8 next_n = (lg_str8){ .len = seq.len, .p = &ctx->text.p[ctx->current_offset]};
    if (lg_strcmp(next_n, seq) == 0) {
        if (out_span != NULL) {
            *out_span = (MRV_Span) {
                .offset = ctx->current_offset,
                .len = seq.len,
            };
        }
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
            if (!mrv_lexer_match_sequence(&ctx, lg_str8_lit("(*"), NULL)) {
                goto single_char;
            }
            mrv_lexer_skip(&ctx);
            mrv_lexer_skip(&ctx);
            const lg_str8 close = lg_str8_lit("*)");
            while (!mrv_lexer_match_sequence(&ctx, close, NULL)) {
                mrv_lexer_skip(&ctx);
            }
            mrv_lexer_skip(&ctx);

            break;
        }

        case '<': {
            MRV_Span span = {0};
            if (!mrv_lexer_match_sequence(&ctx, lg_str8_lit("<<"), &span)) {
                goto unexpected_char;
            }
            mrv_tstream_append(&tstream, artifact_allocator, (MRV_Token){
                .kind = MRV_TokenKind_BeginHostType,
                .span = span,
            });
            break;
        }

        case '>': {
            MRV_Span span = {0};
            const lg_str8 close = lg_str8_lit(">>");
            if (!mrv_lexer_match_sequence(&ctx, close, &span)) {
                goto unexpected_char;
            }
            mrv_tstream_append(&tstream, artifact_allocator, (MRV_Token){
                .kind = MRV_TokenKind_EndHostType,
                .span = span,
            });
            break;
        }

single_char:
        case ')':
        case '{':
        case '}':
        case ':':
        case ',':
        case ';':
        case '=': {
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
    MRV_X(SymbolIdent) \
    MRV_X(HostTypeIdent) \
    MRV_X(OtherIdent) \
    MRV_X(SymbolDeclaration) \
    MRV_X(LanguageDeclaration) \
    MRV_X(StencilDeclaration) \
    MRV_X(TypeDeclaration) \
    MRV_X(OperatorDeclaration) \
    MRV_X(DeclarationArg) \
    MRV_X(DeclarationArgList) \
    MRV_X(InvocationArgList) \
    MRV_X(ExpressionStatement) \
    MRV_X(AssignmentStatement) \
    MRV_X(ControlFlowStatement) \
    MRV_X(ControlFlowDeclaration) \
    MRV_X(ControlFlowBinding) \
    MRV_X(InvocationExpression) \
    MRV_X(Block) \

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


    struct {} SymbolIdent;
    struct {} HostTypeIdent;
    struct {} OtherIdent;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *arg_list;
    } InvocationExpression;

    struct {
        MRV_ASTNode *symbol_declaration;
    } ControlFlowBinding;

    struct {
        MRV_ASTNode *invocation;
        MRV_ASTNode *cf_binding;
        MRV_ASTNode *block;
    } ControlFlowStatement;

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
        MRV_ASTNode *return_type;
    } OperatorDeclaration;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *arg_list;
        MRV_ASTNode *binding_type;
    } ControlFlowDeclaration;

    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *type;
    } DeclarationArg;

    struct {
        size_t         n_args;
        MRV_ASTNode  **args;
    } DeclarationArgList;

    struct {
        size_t         n_args;
        MRV_ASTNode  **args;
    } InvocationArgList;
    
    struct {
        MRV_ASTNode *ident;
        MRV_ASTNode *arg_list;
        MRV_ASTNode *body;
    } StencilDeclaration;

    struct {
        size_t         n_statements;
        MRV_ASTNode  **statements;
    } Block;

    struct {
        MRV_ASTNode *symbol_decl;
        MRV_ASTNode *expression;
    } AssignmentStatement;

    struct {
        MRV_ASTNode *expression;
    } ExpressionStatement;
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
    size_t                 next_offset;

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

lg_force_inline bool
mrv_parser_is_end(MRV_ParserContext *ctx) {
    lg_assert(ctx->cur_block != NULL);

    return (
        ctx->cur_block->next == NULL &&
        ctx->tstream->tail == ctx->cur_block &&
        ctx->next_offset >= ctx->tstream->tail_len
    );
}

lg_force_inline MRV_ParserStatusKind 
mrv_parser_find_next(
    MRV_ParserContext *ctx,
    size_t *out_next_offset,
    MRV_TokenStreamBlock **out_next_block
) {
    lg_assert(ctx != NULL);
    lg_assert(out_next_block != NULL);
    lg_assert(out_next_offset != NULL);

    const size_t cur_len = ctx->cur_block == ctx->tstream->tail ? 
        ctx->tstream->tail_len :
        MRV_TOKEN_STREAM_BLOCK_CAPACITY;

    if (lg_likely(ctx->next_offset < cur_len)) {
        *out_next_block = ctx->cur_block;
        *out_next_offset = ctx->next_offset + 1;
        return MRV_ParserStatusKind_OK;
    }

    if (mrv_parser_is_end(ctx)) {
        return MRV_ParserStatusKind_NOK;
    }

    *out_next_block = ctx->cur_block->next;
    *out_next_offset = 0;

    return MRV_ParserStatusKind_OK;
}

MRV_Token
mrv_parser_consume(MRV_ParserContext *ctx) {
    lg_assert(ctx != NULL);

    size_t next_offset;
    MRV_TokenStreamBlock *next_block;
    MRV_ParserStatusKind status = mrv_parser_find_next(ctx, &next_offset, &next_block);
    lg_assert(status == MRV_ParserStatusKind_OK); // this function shouldn't be called where there isn't a next token
                         // e.g don't expect a token after an EOF
                         // TODO: this assumption may case nodes with variadic children left unterminated
                         // to crash the parser

    MRV_Token ret = ctx->cur_block->tokens[ctx->next_offset];
    ctx->next_offset = next_offset;
    ctx->cur_block = next_block;

    return ret;
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
    return ctx->cur_block->tokens[ctx->next_offset];
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
    if (n_refs == 0) {
        return NULL;
    }

    MRV_ASTNode **refs = lg_arena_alloc_array(&ctx->artifact, MRV_ASTNode*, n_refs);
    lg_assert(refs != NULL);

    uint32_t i = 0;
    while (ctx->ref_stack_top != NULL && i < n_refs) {
        MRV_ASTNode *next = ctx->ref_stack_top->to;
        refs[i] = next;
        ctx->ref_stack_top = ctx->ref_stack_top->prev;
        i++;
    }

    // since they were inserted in stack order, they'll be in reverse-source order
    for (size_t i = 0; i < n_refs / 2; i++) {
        MRV_ASTNode *temp = refs[i];
        refs[i] = refs[n_refs - 1 - i];
        refs[n_refs - 1 - i] = temp;
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
mrv_get_bounding_span(MRV_ParserContext *ctx, size_t n_nodes, MRV_ASTNode **nodes) {
    if (n_nodes == 0) {
        return lg_nil(MRV_Span);
    }

    size_t min_offset = SIZE_MAX,
           max_offset = 0;
    for (size_t i = 0; i < n_nodes; i++) {
        lg_assert(nodes[i] != NULL);

        if (mrv_parser_is_nil_node(ctx, nodes[i])) {
            continue;
        }

        if (nodes[i]->span.offset < min_offset) {
            min_offset = nodes[i]->span.offset;
        }

        size_t offset = nodes[i]->span.offset + nodes[i]->span.len;
        if (offset > max_offset) {
            max_offset = offset;
        }
    }

    return (MRV_Span){ .offset = min_offset, .len = max_offset - min_offset };
}

lg_force_inline void
mrv_parser_unexpected_token(MRV_ParserContext *ctx, MRV_Token tok, MRV_ASTNodeKind parent) {
    mrv_report_error(
        &ctx->err,
        tok.span,
        lg_str8_lit("unexpected token: \"%{str}\" of token kind %{str} in %{str} node"),
        mrv_span_to_str8(tok.span, ctx->text),
        mrv_token_as_str(tok.kind),
        mrv_ast_node_kind_as_str(parent)
    );
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// actual rdp procedures
///
////////////////////////////////////////////////////////////////////////////////

MRV_ASTNode*
mrv_parse_other_ident(MRV_ParserContext *ctx) {
    MRV_Token tok = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, OtherIdent, tok.span);
    return node;
}

MRV_ASTNode*
mrv_parse_host_type_ident(MRV_ParserContext *ctx) {
    MRV_Token ident = mrv_parser_expect(ctx, MRV_TokenKind_Ident);
    mrv_parser_expect(ctx, MRV_TokenKind_EndHostType);
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
mrv_parse_symbol_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *symbol_ident = mrv_parse_symbol_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Colon);
    MRV_ASTNode *type_ident = mrv_parse_other_ident(ctx);

    MRV_Span all_span = mrv_get_bounding_span(ctx, 2, (MRV_ASTNode*[]){symbol_ident, type_ident});
    MRV_ASTNode *node = mrv_parser_mknode(ctx, SymbolDeclaration, all_span, symbol_ident, type_ident);

    return node;
}

MRV_ASTNode*
mrv_parse_invocation_arg_list(MRV_ParserContext *ctx) {
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
            MRV_ASTNode *arg = mrv_parse_other_ident(ctx);
            mrv_parser_nrs_push(ctx, arg);
            n_children++;
            break;
        }

        case MRV_TokenKind_SymbolIdent: {
            MRV_ASTNode *arg = mrv_parse_symbol_ident(ctx);
            mrv_parser_nrs_push(ctx, arg);
            n_children++;
            break;
        }
        
        default:
            mrv_parser_unexpected_token(ctx, peek, MRV_ASTNodeKind_InvocationArgList);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(ctx, n_children, children);
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        InvocationArgList,
        all_span,
        .args = children,
        .n_args = n_children
    );

    lg_pop_scope(&ctx->scratch, scope);

    return node;

}

// forward decl b/c blocks can contain blocks 
MRV_ASTNode*
mrv_parse_block(MRV_ParserContext *ctx);

MRV_ASTNode*
mrv_parse_invocation_expr(MRV_ParserContext *ctx) {
    MRV_ASTNode *op_ident = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);

    MRV_ASTNode *arg_list = mrv_parse_invocation_arg_list(ctx);

    MRV_Span all_span = mrv_get_bounding_span(ctx, 2, (MRV_ASTNode*[]){op_ident, arg_list});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        InvocationExpression,
        all_span,
        .ident = op_ident,
        .arg_list = arg_list,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_control_flow_binding(MRV_ParserContext *ctx) {
    MRV_ASTNode *symbol_decl = mrv_parse_symbol_decl(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_CloseParen);

    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        ControlFlowBinding,
        symbol_decl->span,
        .symbol_declaration = symbol_decl,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_expr_or_cf_stmt(MRV_ParserContext *ctx) {
    MRV_ASTNode *invocation = mrv_parse_invocation_expr(ctx);

    MRV_Token peek = mrv_parser_peek(ctx);
    if (peek.kind == MRV_TokenKind_OpenParen) {
        mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);
        MRV_ASTNode *binding = mrv_parse_control_flow_binding(ctx);

        mrv_parser_expect(ctx, MRV_TokenKind_OpenBrace);
        MRV_ASTNode *block = mrv_parse_block(ctx);
        
        MRV_Span all_span = mrv_get_bounding_span(ctx, 3, (MRV_ASTNode*[]){invocation, binding, block});
        MRV_ASTNode *node = mrv_parser_mknode(
            ctx,
            ControlFlowStatement,
            all_span,
            .invocation = invocation,
            .cf_binding = binding,
            .block = block
        );

        return node;
    }
    
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        ExpressionStatement,
        invocation->span,
        .expression = invocation,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_assignment_statement(MRV_ParserContext *ctx) {
    MRV_ASTNode *symbol_decl = mrv_parse_symbol_decl(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Equals);
    MRV_ASTNode *expr = mrv_parse_invocation_expr(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_Span all_span = mrv_get_bounding_span(ctx, 2, (MRV_ASTNode*[]){symbol_decl, expr});
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        AssignmentStatement,
        all_span,
        .symbol_decl = symbol_decl,
        .expression = expr,
    );
    
    return node; 
}

MRV_ASTNode*
mrv_parse_block(MRV_ParserContext *ctx) {
    LG_Scope scope = lg_push_scope(&ctx->scratch);

    uint32_t n_children = 0;

    while (true) {
        MRV_Token peek = mrv_parser_peek(ctx);
        if (peek.kind ==  MRV_TokenKind_CloseBrace) {
            mrv_parser_consume(ctx);
            goto loop_end;
        }

        switch (peek.kind) {
        case MRV_TokenKind_Ident: {
            MRV_ASTNode *stmt = mrv_parse_expr_or_cf_stmt(ctx);
            mrv_parser_nrs_push(ctx, stmt);
            n_children++;
            break;
        }

        case MRV_TokenKind_SymbolIdent: {
            MRV_ASTNode *stmt = mrv_parse_assignment_statement(ctx);
            mrv_parser_nrs_push(ctx, stmt);
            n_children++;
            break;
        }

        default:
            mrv_parser_unexpected_token(ctx, peek, MRV_ASTNodeKind_Block);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(ctx, n_children, children);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, Block, all_span, .n_statements = n_children, .statements = children);

    lg_pop_scope(&ctx->scratch, scope);
    return node;
}

MRV_ASTNode*
mrv_parse_decl_arg(MRV_ParserContext *ctx) {
    MRV_ASTNode* name = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Colon);

    MRV_ASTNode *type;
    MRV_Token peek = mrv_parser_peek(ctx);
    if (peek.kind == MRV_TokenKind_BeginHostType) {
        mrv_parser_consume(ctx);
        type = mrv_parse_host_type_ident(ctx);
    } else {
        type = mrv_parse_other_ident(ctx);
    }
    
    MRV_Span all_span = mrv_get_bounding_span(ctx, 2, (MRV_ASTNode*[]){name, type});
    MRV_ASTNode *node = mrv_parser_mknode(ctx, DeclarationArg, all_span, .ident = name, .type = type);

    return node;
}

MRV_ASTNode*
mrv_parse_decl_arg_list(MRV_ParserContext *ctx) {
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
            MRV_ASTNode *arg = mrv_parse_decl_arg(ctx);
            mrv_parser_nrs_push(ctx, arg);
            n_children++;
            break;
        }
        default:
            mrv_parser_unexpected_token(ctx, peek, MRV_ASTNodeKind_DeclarationArgList);
            mrv_parser_consume(ctx);
            goto loop_end;
        }
    }
loop_end:;

    MRV_ASTNode **children = mrv_parser_nrs_unwind_cpy(ctx, n_children);
    MRV_Span all_span = mrv_get_bounding_span(ctx, n_children, children);
    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        DeclarationArgList,
        all_span,
        .args = children,
        .n_args = n_children
    );

    lg_pop_scope(&ctx->scratch, scope);

    return node;
}

MRV_ASTNode*
mrv_parse_stencil_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_other_ident(ctx);

    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);
    MRV_ASTNode *arg_list = mrv_parse_decl_arg_list(ctx);

    mrv_parser_expect(ctx, MRV_TokenKind_OpenBrace);
    MRV_ASTNode *body = mrv_parse_block(ctx);

    if (mrv_parser_has_err(ctx)) {
        return mrv_parser_nil_node(ctx);
    }

    MRV_Span all_span = mrv_get_bounding_span(ctx, 3, (MRV_ASTNode*[]){ident, arg_list, body});
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
mrv_parse_language_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, LanguageDeclaration, ident->span, .ident = ident);
    return node;
}

MRV_ASTNode*
mrv_parse_type_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);
    MRV_ASTNode *node = mrv_parser_mknode(ctx, TypeDeclaration, ident->span, .ident = ident);
    return node;
}

MRV_ASTNode*
mrv_parse_operator_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);

    MRV_ASTNode *arg_list = mrv_parse_decl_arg_list(ctx);

    MRV_ASTNode *return_type = mrv_parser_nil_node(ctx);
    MRV_Token peek = mrv_parser_peek(ctx);
    if (peek.kind == MRV_TokenKind_RightArrow) {
        mrv_parser_consume(ctx);
        return_type = mrv_parse_other_ident(ctx);
    }

    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        OperatorDeclaration,
        ident->span,
        .ident = ident,
        .arg_list = arg_list,
        .return_type = return_type,
    );

    return node;
}

MRV_ASTNode*
mrv_parse_control_flow_decl(MRV_ParserContext *ctx) {
    MRV_ASTNode *ident = mrv_parse_other_ident(ctx);
    mrv_parser_expect(ctx, MRV_TokenKind_OpenParen);
    MRV_ASTNode *arg_list = mrv_parse_decl_arg_list(ctx);
    
    MRV_ASTNode *return_type = mrv_parser_nil_node(ctx);
    MRV_Token peek = mrv_parser_peek(ctx);
    if (peek.kind == MRV_TokenKind_RightArrow) {
        mrv_parser_consume(ctx);
        return_type = mrv_parse_other_ident(ctx);
    }

    mrv_parser_expect(ctx, MRV_TokenKind_Semicolon);

    MRV_ASTNode *node = mrv_parser_mknode(
        ctx,
        ControlFlowDeclaration,
        ident->span,
        .ident = ident,
        .arg_list = arg_list,
        .binding_type = return_type,
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

        case MRV_TokenKind_ControlFlow: {
            mrv_parser_consume(ctx);
            MRV_ASTNode *control_flow_decl = mrv_parse_control_flow_decl(ctx);
            mrv_parser_nrs_push(ctx, control_flow_decl);
            break;
        }

        case MRV_TokenKind_Error: 
            lg_unreachable();

        default: {
            mrv_parser_unexpected_token(ctx, tok, MRV_ASTNodeKind_Program);
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
    MRV_Span all_span = mrv_get_bounding_span(ctx, n_children, children);
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
    MRV_TokenStreamBlock *iter_block = ctx.tstream->tail;
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
        case MRV_ASTNodeKind_OtherIdent:
        case MRV_ASTNodeKind_SymbolIdent:
        case MRV_ASTNodeKind_HostTypeIdent:
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
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.arg_list);
            mrv_ast_dump_r(ctx, self, as.OperatorDeclaration.return_type);
            break;

        case MRV_ASTNodeKind_ControlFlowDeclaration:
            mrv_ast_dump_r(ctx, self, as.ControlFlowDeclaration.ident);
            mrv_ast_dump_r(ctx, self, as.ControlFlowDeclaration.arg_list);
            mrv_ast_dump_r(ctx, self, as.ControlFlowDeclaration.binding_type);
            break;

        case MRV_ASTNodeKind_ControlFlowBinding:
            mrv_ast_dump_r(ctx, self, as.ControlFlowBinding.symbol_declaration);
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

        case MRV_ASTNodeKind_InvocationArgList:
            for (uint32_t i = 0; i < as.DeclarationArgList.n_args; i++) {
                mrv_ast_dump_r(ctx, self, as.DeclarationArgList.args[i]);
            }
            break;

        case MRV_ASTNodeKind_InvocationExpression:
            mrv_ast_dump_r(ctx, self, as.InvocationExpression.ident);
            mrv_ast_dump_r(ctx, self, as.InvocationExpression.arg_list);
            break;

        case MRV_ASTNodeKind_ControlFlowStatement:
            mrv_ast_dump_r(ctx, self, as.ControlFlowStatement.invocation);
            mrv_ast_dump_r(ctx, self, as.ControlFlowStatement.cf_binding);
            mrv_ast_dump_r(ctx, self, as.ControlFlowStatement.block);
            break;

        case MRV_ASTNodeKind_StencilDeclaration:
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.ident);
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.arg_list);
            mrv_ast_dump_r(ctx, self, as.StencilDeclaration.body);
            break;

        case MRV_ASTNodeKind_DeclarationArgList:
            for (uint32_t i = 0; i < as.DeclarationArgList.n_args; i++) {
                mrv_ast_dump_r(ctx, self, as.DeclarationArgList.args[i]);
            }
            break;

        case MRV_ASTNodeKind_DeclarationArg:
            mrv_ast_dump_r(ctx, self, as.DeclarationArg.ident);
            mrv_ast_dump_r(ctx, self, as.DeclarationArg.type);
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

        case MRV_ASTNodeKind_AssignmentStatement:
            mrv_ast_dump_r(ctx, self, as.AssignmentStatement.symbol_decl);
            mrv_ast_dump_r(ctx, self, as.AssignmentStatement.expression);
            break;

        case MRV_ASTNodeKind_ExpressionStatement:
            mrv_ast_dump_r(ctx, self, as.ExpressionStatement.expression);
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
MRV_SymbolTable {
    LG_Table  table;
    lg_str8  *types;
} MRV_SymbolTable;

typedef uint8_t
MRV_TypeKind;
enum {
    MRV_TypeKind_Native,
    MRV_TypeKind_Host,
};

typedef uint8_t
MRV_LanguageDescriptorTableEntryKind;
enum {
    MRV_LanguageDescriptorTableEntryKind_Type,
    MRV_LanguageDescriptorTableEntryKind_Operator,
};

typedef struct
MRV_LanguageDescriptorTableEntry {
    MRV_LanguageDescriptorTableEntryKind kind;

    lg_str8 name;

    union {
        struct {
            MRV_TypeKind type_kind;
        } type;

        struct {
            lg_str8  left_arg_name;
            lg_str8  left_arg_type;
            lg_str8  right_arg_type;
            lg_str8  right_arg_name;
            lg_str8  return_type;
        } operator;
    } as;
} MRV_LanguageDescriptorTableEntry;

typedef struct
MRV_LanguageDescriptor {
    lg_str8                            language_name;
    LG_Table                           table;
    MRV_LanguageDescriptorTableEntry  *entries;
    LG_Arena                           arena;
} MRV_LanguageDescriptor;

typedef struct
MRV_SemaContext {
    MRV_AST                *ast;
    lg_str8                 text;
    MRV_SymbolTable         symtab;
    MRV_LanguageDescriptor  ldesc;
    MRV_Error               err;
} MRV_SemaContext;

typedef void (*MRV_SemaVisitor)(MRV_SemaContext *ctx, MRV_ASTNode *self);

lg_force_inline void
mrv_sema_traverse_children(
    MRV_SemaContext *ctx,
    MRV_ASTNode *self,
    MRV_SemaVisitor next
) {
    MRV_ASTNodeChildren as = self->children_as;

    mrv_match_ast_node(self->kind) {
    case MRV_ASTNodeKind_Error:
    case MRV_ASTNodeKind_SymbolIdent:
    case MRV_ASTNodeKind_OtherIdent:
    case MRV_ASTNodeKind_HostTypeIdent:
        break;

    case MRV_ASTNodeKind_Program:
        for (uint32_t i = 0; i < as.Program.n_children; i++) {
            lg_assert(
                as.Program.children[i]->kind == MRV_ASTNodeKind_StencilDeclaration ||
                as.Program.children[i]->kind == MRV_ASTNodeKind_TypeDeclaration ||
                as.Program.children[i]->kind == MRV_ASTNodeKind_OperatorDeclaration ||
                as.Program.children[i]->kind == MRV_ASTNodeKind_ControlFlowDeclaration ||
                as.Program.children[i]->kind == MRV_ASTNodeKind_LanguageDeclaration
            );
            next(ctx, as.Program.children[i]);
        }
        break;

    case MRV_ASTNodeKind_SymbolDeclaration: {
        lg_assert(as.SymbolDeclaration.symbol_ident->kind == MRV_ASTNodeKind_SymbolIdent);
        lg_assert(as.SymbolDeclaration.type_ident->kind == MRV_ASTNodeKind_OtherIdent);

        next(ctx, as.SymbolDeclaration.symbol_ident);
        next(ctx, as.SymbolDeclaration.type_ident);

        break;
    }

    case MRV_ASTNodeKind_InvocationArgList:
        for (size_t i = 0; i < as.InvocationArgList.n_args; i++) {
            lg_assert(
                as.InvocationArgList.args[i]->kind == MRV_ASTNodeKind_OtherIdent ||
                as.InvocationArgList.args[i]->kind == MRV_ASTNodeKind_SymbolIdent
            );
            next(ctx, as.InvocationArgList.args[i]);
        }
        break;

    case MRV_ASTNodeKind_LanguageDeclaration: {
        next(ctx, as.LanguageDeclaration.ident);
        break;
    }

    case MRV_ASTNodeKind_StencilDeclaration: {
        next(ctx, as.StencilDeclaration.ident);
        next(ctx, as.StencilDeclaration.arg_list);
        next(ctx, as.StencilDeclaration.body);
        break;
    }

    case MRV_ASTNodeKind_TypeDeclaration: {
        lg_assert(as.TypeDeclaration.ident->kind == MRV_ASTNodeKind_OtherIdent);
        next(ctx, as.TypeDeclaration.ident);
        break;
    }

    case MRV_ASTNodeKind_OperatorDeclaration: {
        MRV_ASTNode *op_ident = as.OperatorDeclaration.ident;
        MRV_ASTNode *arg_list = as.OperatorDeclaration.arg_list;
        MRV_ASTNode *return_type = as.OperatorDeclaration.return_type;

        bool has_return = !mrv_ast_is_nil_node(ctx->ast, return_type);

        lg_assert(op_ident->kind == MRV_ASTNodeKind_OtherIdent);
        lg_assert(arg_list->kind == MRV_ASTNodeKind_DeclarationArgList);
        lg_assert(!has_return || return_type->kind == MRV_ASTNodeKind_OtherIdent);

        next(ctx, op_ident);
        next(ctx, arg_list);
        next(ctx, return_type);

        break;
    }

    case MRV_ASTNodeKind_ControlFlowDeclaration: {
        MRV_ASTNode *op_ident = as.ControlFlowDeclaration.ident;
        MRV_ASTNode *arg_list = as.ControlFlowDeclaration.arg_list;
        MRV_ASTNode *return_type = as.ControlFlowDeclaration.binding_type;

        bool has_return = !mrv_ast_is_nil_node(ctx->ast, return_type);

        lg_assert(op_ident->kind == MRV_ASTNodeKind_OtherIdent);
        lg_assert(arg_list->kind == MRV_ASTNodeKind_DeclarationArgList);
        lg_assert(!has_return || return_type->kind == MRV_ASTNodeKind_OtherIdent);

        next(ctx, op_ident);
        next(ctx, arg_list);
        next(ctx, return_type);

        break;
    }

    case MRV_ASTNodeKind_DeclarationArg: {
        MRV_ASTNode *ident = as.DeclarationArg.ident;
        MRV_ASTNode *type = as.DeclarationArg.type;

        lg_assert(
            mrv_ast_is_nil_node(ctx->ast, ident) ||
            ident->kind == MRV_ASTNodeKind_OtherIdent
        );
        lg_assert(
            mrv_ast_is_nil_node(ctx->ast, type) ||
            type->kind == MRV_ASTNodeKind_OtherIdent ||
            type->kind == MRV_ASTNodeKind_HostTypeIdent
        );

        next(ctx, ident);
        next(ctx, type);

        break;
    }
        
    case MRV_ASTNodeKind_InvocationExpression: {
        MRV_ASTNode *ident = as.InvocationExpression.ident;
        MRV_ASTNode *arg_list = as.InvocationExpression.arg_list;

        lg_assert(ident->kind == MRV_ASTNodeKind_OtherIdent);
        lg_assert(arg_list->kind == MRV_ASTNodeKind_InvocationArgList);

        next(ctx, ident);
        next(ctx, arg_list);

        break;
    }

    case MRV_ASTNodeKind_ControlFlowStatement: {
        MRV_ASTNode *invocation = as.ControlFlowStatement.invocation;
        MRV_ASTNode *block = as.ControlFlowStatement.block;
        MRV_ASTNode *binding = as.ControlFlowStatement.cf_binding;

        lg_assert(invocation->kind == MRV_ASTNodeKind_InvocationExpression);
        lg_assert(block->kind == MRV_ASTNodeKind_Block);
        lg_assert(binding->kind == MRV_ASTNodeKind_ControlFlowBinding);

        next(ctx, invocation);
        next(ctx, block);
        next(ctx, binding);

        break;
    }

    case MRV_ASTNodeKind_ControlFlowBinding: {
        lg_assert(as.ControlFlowBinding.symbol_declaration->kind == MRV_ASTNodeKind_SymbolDeclaration);
        next(ctx, as.ControlFlowBinding.symbol_declaration);
        break;
    }

    case MRV_ASTNodeKind_DeclarationArgList:
        for (uint32_t i = 0; i < as.DeclarationArgList.n_args; i++) {
            lg_assert(as.DeclarationArgList.args[i]->kind == MRV_ASTNodeKind_DeclarationArg);
            next(ctx, as.DeclarationArgList.args[i]);
        }
        break;
        
    case MRV_ASTNodeKind_Block:
        for (uint32_t i = 0; i < as.Block.n_statements; i++) {
            lg_assert(
                as.Block.statements[i]->kind == MRV_ASTNodeKind_ControlFlowStatement ||
                as.Block.statements[i]->kind == MRV_ASTNodeKind_AssignmentStatement ||
                as.Block.statements[i]->kind == MRV_ASTNodeKind_ExpressionStatement
            );
            next(ctx, as.Block.statements[i]);
        }
        break;

    case MRV_ASTNodeKind_AssignmentStatement:
        next(ctx, as.AssignmentStatement.symbol_decl);
        next(ctx, as.AssignmentStatement.expression);
        break;

    case MRV_ASTNodeKind_ExpressionStatement:
        next(ctx, as.ExpressionStatement.expression);
        break;
    }
}

void
mrv_sema_record_type_decls_r(MRV_SemaContext *ctx, MRV_ASTNode *self) {
    if (mrv_ast_is_nil_node(ctx->ast, self)) {
        return;
    }

    mrv_match_ast_node(self->kind) {
    case MRV_ASTNodeKind_Program:
    case MRV_ASTNodeKind_OperatorDeclaration:
    case MRV_ASTNodeKind_InvocationExpression:
    case MRV_ASTNodeKind_StencilDeclaration: 
    case MRV_ASTNodeKind_DeclarationArg:
    case MRV_ASTNodeKind_DeclarationArgList: {
        mrv_sema_traverse_children(ctx, self, mrv_sema_record_type_decls_r);
        break;
    }

    case MRV_ASTNodeKind_HostTypeIdent: {
        lg_str8 ident = mrv_span_to_str8(self->span, ctx->text);

        size_t idx;
        LG_StatusKind status = lg_table_ensure_str8(
            &ctx->ldesc.table,
            ident,
            &idx,
            NULL
        );
        lg_assert(status == LG_StatusKind_OK);

        ctx->ldesc.entries[idx].name = ident;
        ctx->ldesc.entries[idx].kind = MRV_LanguageDescriptorTableEntryKind_Type;
        ctx->ldesc.entries[idx].as.type.type_kind = MRV_TypeKind_Host;

        break;
    }

    // we'll also take this opportunity to record the language name
    case MRV_ASTNodeKind_LanguageDeclaration: {
        lg_str8 language_name = mrv_span_to_str8(self->children_as.LanguageDeclaration.ident->span, ctx->text);
        if (
            ctx->ldesc.language_name.len != 0 && 
            (lg_strcmp(language_name, ctx->ldesc.language_name) != 0)
        ) {
            mrv_report_error(&ctx->err, self->span, lg_str8_lit("conflicting langauge declarations found"));
        }
        ctx->ldesc.language_name = language_name;
        break;
    }

    case MRV_ASTNodeKind_TypeDeclaration: {
        lg_str8 ident = mrv_span_to_str8(self->children_as.TypeDeclaration.ident->span, ctx->text);

        size_t idx;
        bool found;
        LG_StatusKind status = lg_table_ensure_str8(
            &ctx->ldesc.table,
            ident,
            &idx,
            &found
        );
        if (found) {
            mrv_report_error(
                &ctx->err,
                self->span,
                lg_str8_lit("type %{str} declared multiple times"),
                ident
            );
            break;
        }
        lg_assert(status == LG_StatusKind_OK);

        ctx->ldesc.entries[idx].name = ident;
        ctx->ldesc.entries[idx].kind = MRV_LanguageDescriptorTableEntryKind_Type;
        ctx->ldesc.entries[idx].as.type.type_kind = MRV_TypeKind_Native;

        break;
    }

    default:;
    }
}

void
mrv_sema_record_op_and_cf_decls_r(MRV_SemaContext *ctx, MRV_ASTNode *self) {
    if (mrv_ast_is_nil_node(ctx->ast, self)) {
        return;
    }

    MRV_ASTNodeChildren as = self->children_as;

    mrv_match_ast_node(self->kind) {
    case MRV_ASTNodeKind_Program:
        mrv_sema_traverse_children(ctx, self, mrv_sema_record_op_and_cf_decls_r);
        break;

    case MRV_ASTNodeKind_OperatorDeclaration: {
        MRV_ASTNode *op = as.OperatorDeclaration.ident;
        MRV_ASTNode *arg_list = as.OperatorDeclaration.arg_list;
        MRV_ASTNode *return_type = as.OperatorDeclaration.return_type;

        bool has_return = !mrv_ast_is_nil_node(ctx->ast, return_type);

        lg_str8 op_ident = mrv_span_to_str8(op->span, ctx->text);
        lg_str8 return_type_ident = has_return ?
            mrv_span_to_str8(return_type->span, ctx->text) :
            lg_nil(lg_str8);

        lg_str8 arg_names[2] = {0};
        lg_str8 arg_types[2] = {0};
        {
            size_t n_args = arg_list->children_as.DeclarationArgList.n_args;
            if (n_args > 2) {
                mrv_report_error(
                    &ctx->err,
                    arg_list->span,
                    lg_str8_lit(
                        "operator %{str} declared with %{i64} arguments\n"
                        "operators may not have more than two arguments"
                    ),
                    op_ident, n_args
                );
            }

            bool found;
            
            for (size_t i = 0; i < n_args; i++) {
                lg_assert(i < 2);
                MRV_Span name_ident_span = arg_list->children_as.DeclarationArgList.args[i]->children_as.DeclarationArg.ident->span;
                lg_str8 name_ident = mrv_span_to_str8(name_ident_span, ctx->text);
                MRV_Span type_ident_span = arg_list->children_as.DeclarationArgList.args[i]->children_as.DeclarationArg.type->span;
                lg_str8 type_ident = mrv_span_to_str8(type_ident_span, ctx->text);

                size_t idx = lg_table_get_str8(&ctx->ldesc.table, type_ident, &found);
                if (!found) {
                    mrv_report_error(
                        &ctx->err,
                        type_ident_span,
                        lg_str8_lit("unknown type in args of operator declaration: %{str}"),
                        type_ident
                    );
                    break;
                }
                if (ctx->ldesc.entries[idx].kind != MRV_LanguageDescriptorTableEntryKind_Type) {
                    mrv_report_error(
                        &ctx->err,
                        type_ident_span,
                        lg_str8_lit("type %{str} in args of operator declaration is not a type at all"),
                        type_ident
                    );
                    break;
                }

                arg_names[i] = name_ident;
                arg_types[i] = type_ident;
            }

            if (has_return) {
                size_t idx = lg_table_get_str8(&ctx->ldesc.table, return_type_ident, &found);
                if (!found) {
                    mrv_report_error(
                        &ctx->err,
                        return_type->span,
                        lg_str8_lit("unknown type in return type of operator declaration: %{str}"),
                        return_type_ident
                    );
                    break;
                }
                if (ctx->ldesc.entries[idx].kind != MRV_LanguageDescriptorTableEntryKind_Type) {
                    mrv_report_error(
                        &ctx->err,
                        return_type->span,
                        lg_str8_lit("type %{str} in args of operator declaration is not a type at all"),
                        return_type_ident
                    );
                    break;
                }
            }
        }

        size_t op_idx;
        {
            bool found;
            LG_StatusKind status = lg_table_ensure_str8(&ctx->ldesc.table, op_ident, &op_idx, &found);
            lg_assert(status == LG_StatusKind_OK);
            if (found) {
                mrv_report_error(
                    &ctx->err,
                    self->span,
                    lg_str8_lit("multiple declarations found for operator %{str}"),
                    op_ident
                );
                break;
            }
        }

        ctx->ldesc.entries[op_idx].name = op_ident;
        ctx->ldesc.entries[op_idx].kind = MRV_LanguageDescriptorTableEntryKind_Operator;
        ctx->ldesc.entries[op_idx].as.operator.left_arg_name = arg_names[0];
        ctx->ldesc.entries[op_idx].as.operator.left_arg_type = arg_types[0];
        ctx->ldesc.entries[op_idx].as.operator.right_arg_name = arg_names[1];
        ctx->ldesc.entries[op_idx].as.operator.right_arg_type = arg_types[1];
        ctx->ldesc.entries[op_idx].as.operator.return_type = return_type_ident;

        break;
    }

    case MRV_ASTNodeKind_ControlFlowDeclaration: {
        LG_StatusKind status = LG_StatusKind_OK;

        MRV_ASTNode *feature_name = as.ControlFlowDeclaration.ident;
        MRV_ASTNode *arg_list = as.ControlFlowDeclaration.arg_list;
        MRV_ASTNode *binding_type = as.ControlFlowDeclaration.binding_type;

        bool has_binding = !mrv_ast_is_nil_node(ctx->ast, binding_type);

        lg_str8 feature_name_ident = mrv_span_to_str8(feature_name->span, ctx->text);
        lg_str8 binding_type_ident = has_binding ?
            mrv_span_to_str8(binding_type->span, ctx->text) :
            lg_nil(lg_str8);

        lg_str8 arg_name = {0};
        lg_str8 arg_type = {0};
        {
            size_t n_args = arg_list->children_as.DeclarationArgList.n_args;
            if (n_args > 1) {
                mrv_report_error(
                    &ctx->err,
                    arg_list->span,
                    lg_str8_lit(
                        "control flow feature %{str} declared with %{i64} arguments\n"
                        "control flow features may not have more than one argument"
                    ),
                    feature_name_ident, n_args
                );
            }

            if (n_args > 0) {
                bool found;
            
                MRV_Span name_ident_span = arg_list->children_as.DeclarationArgList.args[0]->children_as.DeclarationArg.ident->span;
                lg_str8 name_ident = mrv_span_to_str8(name_ident_span, ctx->text);
                MRV_Span type_ident_span = arg_list->children_as.DeclarationArgList.args[0]->children_as.DeclarationArg.type->span;
                lg_str8 type_ident = mrv_span_to_str8(type_ident_span, ctx->text);

                size_t idx = lg_table_get_str8(&ctx->ldesc.table, type_ident, &found);
                if (!found) {
                    mrv_report_error(
                        &ctx->err,
                        type_ident_span,
                        lg_str8_lit("unknown type in args of control flow declaration: %{str}"),
                        type_ident
                    );
                    break;
                }
                if (ctx->ldesc.entries[idx].kind != MRV_LanguageDescriptorTableEntryKind_Type) {
                    mrv_report_error(
                        &ctx->err,
                        type_ident_span,
                        lg_str8_lit("type %{str} in args of control flow declaration is not a type at all"),
                        type_ident
                    );
                    break;
                }

                arg_name = name_ident;
                arg_type = type_ident;

                if (has_binding) {
                    size_t idx = lg_table_get_str8(&ctx->ldesc.table, binding_type_ident, &found);
                    if (!found) {
                        mrv_report_error(
                            &ctx->err,
                            binding_type->span,
                            lg_str8_lit("unknown type in binding type of control flow declaration: %{str}"),
                            binding_type_ident
                        );
                        break;
                    }
                    if (ctx->ldesc.entries[idx].kind != MRV_LanguageDescriptorTableEntryKind_Type) {
                        mrv_report_error(
                            &ctx->err,
                            binding_type->span,
                            lg_str8_lit("type %{str} in args of control flow declaration is not a type at all"),
                            binding_type_ident
                        );
                        break;
                    }
                }
            }
        }

        // this is the point where we actually split control flow features into two parts:
        // 1) the "begin" operator, which creates the binding variable, and
        // 2) the "end" operator, which consumes said binding varialble.
        //
        // the downside of doing it here is that you have to treat the end operators as a special
        // case during scope resolution.
        //
        // the upside, which in this case bears much more weight, is you get to delete a tone of code
        // for dealing with control flow features during code generation.

        lg_str8 begin_name = {0};
        status = lg_strcat(&ctx->ldesc.arena, (lg_str8[]){feature_name_ident, lg_str8_lit("Begin")}, 2, &begin_name);
        lg_assert(status == LG_StatusKind_OK);

        lg_str8 end_name = {0};
        status = lg_strcat(&ctx->ldesc.arena, (lg_str8[]){feature_name_ident, lg_str8_lit("End")}, 2, &end_name);
        lg_assert(status == LG_StatusKind_OK);

        size_t begin_idx;
        size_t end_idx;
        {
            bool found;
            status = lg_table_ensure_str8(&ctx->ldesc.table, begin_name, &begin_idx, &found);
            lg_assert(status == LG_StatusKind_OK);
            if (found) {
                mrv_report_error(
                    &ctx->err,
                    self->span,
                    lg_str8_lit("multiple declarations found for control flow feature %{str}"),
                    feature_name_ident
                );
                break;
            }

            status = lg_table_ensure_str8(&ctx->ldesc.table, end_name, &end_idx, &found);
            lg_assert(status == LG_StatusKind_OK);
            lg_assert(!found);
        }

        ctx->ldesc.entries[begin_idx].name = begin_name;
        ctx->ldesc.entries[begin_idx].kind = MRV_LanguageDescriptorTableEntryKind_Operator;
        ctx->ldesc.entries[begin_idx].as.operator.left_arg_name = arg_name;
        ctx->ldesc.entries[begin_idx].as.operator.left_arg_type = arg_type;
        ctx->ldesc.entries[begin_idx].as.operator.return_type = binding_type_ident;

        ctx->ldesc.entries[end_idx].name = end_name;
        ctx->ldesc.entries[end_idx].kind = MRV_LanguageDescriptorTableEntryKind_Operator;
        ctx->ldesc.entries[end_idx].as.operator.left_arg_name = lg_str8_lit("binding");
        ctx->ldesc.entries[end_idx].as.operator.left_arg_type = binding_type_ident;

        break;
    }

    default:;
    }
}

void
mrv_analyze(
    LG_Allocator *artifact_allocator,
    LG_Allocator *scratch_allocator,
    MRV_AST *ast,
    lg_str8 text,
    LG_Writer *err_writer,
    MRV_LanguageDescriptor *out_ldesc
) {
    lg_assert(out_ldesc != NULL);

    LG_Arena arena = {0};
    lg_arena_init(&arena, scratch_allocator);
    
    MRV_SemaContext ctx = {
        .ast = ast,
        .text = text,
        .err.writer = err_writer,
    };


    //////////////////////////////////////////////
    /// ~~ initialize tables ~~
    // TODO: remove magic number capacity

    LG_StatusKind status = LG_StatusKind_OK;

    status = lg_table_init(&ctx.symtab.table, &arena, 1024);
    lg_assert(status == LG_StatusKind_OK);
    ctx.symtab.types = lg_arena_alloc_array(&arena, lg_str8, 1024);
    lg_assert(ctx.symtab.types != NULL);

    lg_arena_init(&ctx.ldesc.arena, artifact_allocator);
    status = lg_table_init(&ctx.ldesc.table, &ctx.ldesc.arena, 1024);
    lg_assert(status == LG_StatusKind_OK);
    ctx.ldesc.entries = lg_arena_alloc_array(&ctx.ldesc.arena, MRV_LanguageDescriptorTableEntry, 1024);
    lg_assert(ctx.ldesc.entries != NULL);


    //////////////////////////////////////////////
    /// ~~ do the type checking ~~

    mrv_sema_record_type_decls_r(&ctx, ctx.ast->root);
    mrv_sema_record_op_and_cf_decls_r(&ctx, ctx.ast->root);

    
    //////////////////////////////////////////////
    /// ~~ fin ~~
    
    *out_ldesc = ctx.ldesc;


    lg_arena_free_all(&arena);
}

void
mrv_ldesc_destroy(MRV_LanguageDescriptor *ldesc) {
    lg_arena_free_all(&ldesc->arena);
    lg_memzero(ldesc, sizeof(MRV_LanguageDescriptor));
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// text templates
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_TmplFieldTable {
    lg_str8 key;
    union {
        lg_str8 str;
        LG_StringList strlist;
    } value_as;
} MRV_TmplFieldTable;

void 
mrv_write_tmpl(
    LG_Writer *writer,
    lg_str8 text,
    MRV_TmplFieldTable *fields,
    size_t n_entries
) {
    for (size_t i = 0 ; i < text.len; i++) {
        if (
            text.p[i] == '$' &&
            (i + 1 < text.len && text.p[i + 1] == '{') &&
            (i + 2 < text.len && text.p[i + 2] == '{')
        ) {
            i += 2;

            size_t scan = i;
            while (
                (scan < text.len && text.p[scan] != '}') &&
                (scan + 1 < text.len && text.p[scan + 1] != '}')
            ) { scan++; }

            lg_str8 found_key = (lg_str8){ .len = scan - i, .p = text.p + i + 1 };

            if (
                found_key.len > 2 &&
                found_key.p[0] == 'L' &&
                found_key.p[1] == ':'
            ) {
                bool found = false;
                lg_str8 found_key_without_prefix = (lg_str8){ .len = found_key.len - 2, .p = found_key.p + 2 };
                LG_StringList found_value = {0};
                for (size_t i_table = 0; i_table < n_entries; i_table++) {
                    if (lg_strcmp(found_key_without_prefix, fields[i_table].key) == 0) {
                        found = true;
                        found_value = fields[i_table].value_as.strlist;
                        break;
                    }
                }

                lg_assert(found);
                lg_strlist_write(&found_value, writer);
            } else {
                bool found = false;
                lg_str8 found_value = {0};
                for (size_t i_table = 0; i_table < n_entries; i_table++) {
                    if (lg_strcmp(found_key, fields[i_table].key) == 0) {
                        found = true;
                        found_value = fields[i_table].value_as.str;
                        break;
                    }
                }

                lg_assert(found);
                lg_write(writer, found_value);
            }

            i = scan + 2;
        } else {
            lg_write(writer, ((lg_str8){ .len = 1, .p = text.p + i }));
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// the sourcegen stuff
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
MRV_SourcegenContext {
    LG_Arena                *scratch;
    LG_Writer               *header_file_writer;
    MRV_LanguageDescriptor  *ldesc;
    lg_str8                  type_ident_prefix;

    struct {
        lg_str8 lang_capitalized;
        lg_str8 lang_snake_case;
    } common_strings;
} MRV_SourcegenContext;

void
mrv_sg_symbol_type(MRV_SourcegenContext *ctx, lg_str8 name) {
    bool found;
    size_t idx = lg_table_get_str8(&ctx->ldesc->table, name, &found);
    lg_assert(found);

    MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];

    lg_assert(entry.kind == MRV_LanguageDescriptorTableEntryKind_Type);

    if (entry.as.type.type_kind == MRV_TypeKind_Native) {
        lg_printf(
            ctx->header_file_writer,
            lg_str8_lit("LG_%{str}Symbol_%{str}"),
            ctx->ldesc->language_name, entry.name
        );
    } else if (entry.as.type.type_kind == MRV_TypeKind_Host) {
        lg_write(ctx->header_file_writer, entry.name);
    } else {
        lg_unreachable();
    }
}

void
mrv_sg_type_enum(MRV_SourcegenContext *ctx) {
    lg_printf(ctx->header_file_writer, lg_str8_lit(
        "\ntypedef uint8_t\nLG_%{str}Type;\n"
        "enum\nLG_%{str}Type {\n"
    ), ctx->ldesc->language_name, ctx->ldesc->language_name);

    LG_TableIter iter = {0};
    lg_table_iter_init(&iter, &ctx->ldesc->table);

    size_t idx;
    while (lg_table_iter_advance(&iter, &idx, NULL)) {
        MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];
        if (
            entry.kind != MRV_LanguageDescriptorTableEntryKind_Type ||
            entry.as.type.type_kind != MRV_TypeKind_Native
        ) {
            continue;
        }

        lg_printf(
            ctx->header_file_writer,
            lg_str8_lit("    LG_%{str}Type_%{str},\n"),
            ctx->ldesc->language_name, entry.name
        );
    }

    lg_write(ctx->header_file_writer, lg_str8_lit("};\n"));
}

void
mrv_sg_opcode_enum(MRV_SourcegenContext *ctx) {
    lg_printf(ctx->header_file_writer, lg_str8_lit(
        "\ntypedef uint8_t\nLG_%{str}Opcode;\n"
        "enum\nLG_%{str}Opcode {\n"
    ), ctx->ldesc->language_name, ctx->ldesc->language_name);

    LG_TableIter iter = {0};
    lg_table_iter_init(&iter, &ctx->ldesc->table);

    size_t idx;
    while (lg_table_iter_advance(&iter, &idx, NULL)) {
        MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];
        if (entry.kind == MRV_LanguageDescriptorTableEntryKind_Operator) {
            lg_printf(
                ctx->header_file_writer,
                lg_str8_lit("    LG_%{str}Opcode_%{str},\n"),
                ctx->ldesc->language_name, entry.name
            );
        }
    }

    lg_write(ctx->header_file_writer, lg_str8_lit("};\n"));
}

void
mrv_sg_symbol_types(MRV_SourcegenContext *ctx) {
    LG_TableIter iter = {0};
    lg_table_iter_init(&iter, &ctx->ldesc->table);

    size_t idx;
    while (lg_table_iter_advance(&iter, &idx, NULL)) {
        MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];
        if (
            entry.kind != MRV_LanguageDescriptorTableEntryKind_Type ||
            entry.as.type.type_kind != MRV_TypeKind_Native
        ) {
            continue;
        }

        lg_write(ctx->header_file_writer, lg_str8_lit("\ntypedef struct\n"));
        mrv_sg_symbol_type(ctx, entry.name);
        lg_write(ctx->header_file_writer, lg_str8_lit(" {\n    uint32_t id;\n} "));
        mrv_sg_symbol_type(ctx, entry.name);
        lg_write(ctx->header_file_writer, lg_str8_lit(";\n"));
    }
}

void
mrv_sg_node_types(MRV_SourcegenContext *ctx) {
    LG_TableIter iter = {0};
    lg_table_iter_init(&iter, &ctx->ldesc->table);

    size_t idx;
    while (lg_table_iter_advance(&iter, &idx, NULL)) {
        MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];
        if (entry.kind == MRV_LanguageDescriptorTableEntryKind_Operator) {
            lg_printf(
                ctx->header_file_writer,
                lg_str8_lit("\ntypedef struct\nLG_%{str}Node_%{str} {"),
                ctx->ldesc->language_name, entry.name
            );

            if (entry.as.operator.left_arg_name.len != 0) {
                lg_write(ctx->header_file_writer, lg_str8_lit("\n    "));
                mrv_sg_symbol_type(ctx, entry.as.operator.left_arg_type);
                lg_printf(
                    ctx->header_file_writer,
                    lg_str8_lit(" %{str};"),
                    entry.as.operator.left_arg_name
                );
            }
            if (entry.as.operator.right_arg_name.len != 0) {
                lg_write(ctx->header_file_writer, lg_str8_lit("\n    "));
                mrv_sg_symbol_type(ctx, entry.as.operator.right_arg_type);
                lg_printf(
                    ctx->header_file_writer,
                    lg_str8_lit(" %{str};"),
                    entry.as.operator.right_arg_name
                );
            }
            if (entry.as.operator.return_type.len != 0) {
                lg_write(ctx->header_file_writer, lg_str8_lit("\n    "));
                mrv_sg_symbol_type(ctx, entry.as.operator.return_type);
                lg_write(ctx->header_file_writer, lg_str8_lit(" return_val;"));
            }
            
            lg_printf(
                ctx->header_file_writer,
                lg_str8_lit("\n} LG_%{str}Node_%{str};\n"),
                ctx->ldesc->language_name, entry.name
            );
        }
    }
}

void
mrv_sg_node_union_type(MRV_SourcegenContext *ctx) {
    lg_printf(ctx->header_file_writer, lg_str8_lit("\ntypedef union\nLG_%{str}Operands {"), ctx->ldesc->language_name);
    {
        LG_TableIter iter = {0};
        lg_table_iter_init(&iter, &ctx->ldesc->table);

        size_t idx;
        while (lg_table_iter_advance(&iter, &idx, NULL)) {
            LG_Scope scope = lg_push_scope(ctx->scratch);

            MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];

            lg_str8 name_snake_case;
            LG_StatusKind status = lg_str8_pascal_to_snake_case(entry.name, ctx->scratch, &name_snake_case);
            lg_assert(status == LG_StatusKind_OK);

            if (entry.kind == MRV_LanguageDescriptorTableEntryKind_Operator) {
                lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}Node_%{str} %{str};"), ctx->ldesc->language_name, entry.name, name_snake_case);
            }

            lg_pop_scope(ctx->scratch, scope);
        }
    }
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n} LG_%{str}Operands;\n"), ctx->ldesc->language_name);

    lg_printf(ctx->header_file_writer, lg_str8_lit("\ntypedef struct\nLG_%{str}Node {"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}Opcode opcode;"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}Operands operands_as;"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n} LG_%{str}Node;\n"), ctx->ldesc->language_name);
}

void
mrv_sg_builder_types(MRV_SourcegenContext *ctx) {
    lg_printf(ctx->header_file_writer, lg_str8_lit("\ntypedef struct\nLG_%{str}NodeList {"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    struct LG_%{str}NodeList *prev;"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}Node node;"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n} LG_%{str}NodeList;\n"), ctx->ldesc->language_name);

    lg_printf(ctx->header_file_writer, lg_str8_lit("\ntypedef struct\nLG_%{str}Builder {"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}NodeList *nodes_tail;"), ctx->ldesc->language_name);
    lg_write(ctx->header_file_writer, lg_str8_lit("\n    uint32_t next_symbol_id;"));
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n} LG_%{str}Builder;\n"), ctx->ldesc->language_name);
}

void
mrv_sg_expr_type(MRV_SourcegenContext *ctx) {
    lg_printf(ctx->header_file_writer, lg_str8_lit("\ntypedef struct\nLG_%{str}Expr {"), ctx->ldesc->language_name);
    lg_write(ctx->header_file_writer, lg_str8_lit("\n    size_t cap;"));
    lg_write(ctx->header_file_writer, lg_str8_lit("\n    size_t len;"));
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n    LG_%{str}Node *nodes;"), ctx->ldesc->language_name);
    lg_printf(ctx->header_file_writer, lg_str8_lit("\n} LG_%{str}Expr;\n"), ctx->ldesc->language_name);
}

void
mrv_sg_append_fn(MRV_SourcegenContext *ctx) {
    const lg_str8 template = lg_str8_lit(R"(
lg_force_inline ${{L:return_type}}
lg_hbuilder_${{op_snake}}(
    LG_Context *ctx,
    LG_${{lang_name}}Builder *builder${{L:operands}}
) {
    LG_${{lang_name}}NodeList *node = lg_arena_alloc_struct(&ctx->arena, LG_${{lang_name}}NodeList);
    if (node == NULL) {
        lg_report_error(ctx, LG_StatusKind_OutOfMemory, lg_str8_lit("ran out of memory appending to ${{lang_snake}} expr"));
        return 0;
    }

    // increment first so zero is not a valid symbol id
    builder->next_symbol_id++;
    uint32_t y = next_symbol_id;

    if (builder->ir_tail != NULL) {
        node->prev = builder->ir_tail;
    }
    builder->ir_tail = node;

    return y;
}
)");
    
    LG_TableIter iter = {0};
    lg_table_iter_init(&iter, &ctx->ldesc->table);

    size_t idx;
    while (lg_table_iter_advance(&iter, &idx, NULL)) {
        MRV_LanguageDescriptorTableEntry entry = ctx->ldesc->entries[idx];

        if (entry.kind != MRV_LanguageDescriptorTableEntryKind_Operator) {
            continue;
        }

        LG_Scope scope = lg_push_scope(ctx->scratch);
        LG_StatusKind status = LG_StatusKind_OK;

        lg_str8 name_snake;
        status = lg_str8_pascal_to_snake_case(entry.name, ctx->scratch, &name_snake);
        lg_assert(status == LG_StatusKind_OK);

        LG_StringList operands = {0};

        if (entry.as.operator.left_arg_name.len > 0) {
            lg_str8 arg_name_snake = {0};
            status = lg_str8_pascal_to_snake_case(entry.as.operator.left_arg_name, ctx->scratch, &arg_name_snake);

            bool found;
            size_t arg_idx = lg_table_get_str8(&ctx->ldesc->table, entry.as.operator.left_arg_type, &found);
            lg_assert(found);
            lg_assert(ctx->ldesc->entries[arg_idx].kind == MRV_LanguageDescriptorTableEntryKind_Type);

            lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit(",\n    "));

            if (ctx->ldesc->entries[arg_idx].as.type.type_kind == MRV_TypeKind_Native) {
                lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit("LG_"));
                lg_strlist_cpy_append(&operands, ctx->scratch, ctx->ldesc->language_name);
                lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit("Symbol_"));
                lg_strlist_cpy_append(&operands, ctx->scratch, entry.as.operator.left_arg_type);
            } else if (ctx->ldesc->entries[arg_idx].as.type.type_kind == MRV_TypeKind_Host) {
                lg_strlist_cpy_append(&operands, ctx->scratch, entry.as.operator.left_arg_type);
            } else {
                lg_unreachable();
            }

            lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit(" "));
            lg_strlist_cpy_append(&operands, ctx->scratch, arg_name_snake);
        }
        if (entry.as.operator.right_arg_name.len > 0) {
            lg_str8 arg_name_snake = {0};
            status = lg_str8_pascal_to_snake_case(entry.as.operator.right_arg_name, ctx->scratch, &arg_name_snake);

            bool found;
            size_t arg_idx = lg_table_get_str8(&ctx->ldesc->table, entry.as.operator.right_arg_type, &found);
            lg_assert(found);
            lg_assert(ctx->ldesc->entries[arg_idx].kind == MRV_LanguageDescriptorTableEntryKind_Type);

            lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit(",\n    "));

            if (ctx->ldesc->entries[arg_idx].as.type.type_kind == MRV_TypeKind_Native) {
                lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit("LG_"));
                lg_strlist_cpy_append(&operands, ctx->scratch, ctx->ldesc->language_name);
                lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit("Symbol_"));
                lg_strlist_cpy_append(&operands, ctx->scratch, entry.as.operator.right_arg_type);
            } else if (ctx->ldesc->entries[arg_idx].as.type.type_kind == MRV_TypeKind_Host) {
                lg_strlist_cpy_append(&operands, ctx->scratch, entry.as.operator.right_arg_type);
            } else {
                lg_unreachable();
            }

            lg_strlist_cpy_append(&operands, ctx->scratch, lg_str8_lit(" "));
            lg_strlist_cpy_append(&operands, ctx->scratch, arg_name_snake);
        }

        LG_StringList return_type = {0};
        if (entry.as.operator.return_type.len > 0) {
            lg_strlist_cpy_append(&return_type, ctx->scratch, lg_str8_lit("LG_"));
            lg_strlist_cpy_append(&return_type, ctx->scratch, ctx->ldesc->language_name);
            lg_strlist_cpy_append(&return_type, ctx->scratch, lg_str8_lit("Symbol_"));
            lg_strlist_cpy_append(&return_type, ctx->scratch, entry.as.operator.return_type);
        } else {
            lg_strlist_cpy_append(&return_type, ctx->scratch, lg_str8_lit("void"));
        }

        const size_t len = 5;
        MRV_TmplFieldTable fields[5] = {
            {lg_str8_lit("lang_name"),    { .str = ctx->ldesc->language_name }},
            {lg_str8_lit("lang_snake"),   { .str = ctx->common_strings.lang_snake_case}},
            {lg_str8_lit("return_type"),  { .strlist = return_type }},
            {lg_str8_lit("op_snake"),     { .str = name_snake }},
            {lg_str8_lit("operands"),     { .strlist = operands }},
        };
        mrv_write_tmpl(ctx->header_file_writer, template, fields, len);

        lg_pop_scope(ctx->scratch, scope);
    }
}

void
mrv_gen_source(
    LG_Writer *header_file_writer,
    LG_Arena *scratch_allocator,
    MRV_LanguageDescriptor *ldesc
) {
    MRV_SourcegenContext ctx = {
        .header_file_writer = header_file_writer,
        .scratch = scratch_allocator,
        .ldesc = ldesc,
    };

    LG_Scope scope = lg_push_scope(ctx.scratch);

    // common strigs
    {
        LG_StatusKind status = LG_StatusKind_OK;

        status = lg_str8_to_upper(ldesc->language_name, ctx.scratch, &ctx.common_strings.lang_capitalized);
        status = lg_str8_pascal_to_snake_case(ldesc->language_name, ctx.scratch, &ctx.common_strings.lang_snake_case);

        lg_assert(status == LG_StatusKind_OK);
    }

    lg_printf(
        header_file_writer, 
        lg_str8_lit(
            "#ifndef LG_%{str}_GEN_H_\n"
            "#define LG_%{str}_GEN_H_\n"
        ), 
        ctx.common_strings.lang_capitalized,
        ctx.common_strings.lang_capitalized
    );
    lg_write(header_file_writer, lg_str8_lit("\n#include <libgrad/internal/base.h>\n"));

    mrv_sg_type_enum(&ctx);
    mrv_sg_opcode_enum(&ctx);
    mrv_sg_symbol_types(&ctx);
    mrv_sg_node_types(&ctx);
    mrv_sg_node_union_type(&ctx);
    mrv_sg_builder_types(&ctx);
    mrv_sg_expr_type(&ctx);
    mrv_sg_append_fn(&ctx);

    lg_printf(header_file_writer, lg_str8_lit("\n#endif // LG_%{str}_GEN_H_\n"), ctx.common_strings.lang_capitalized);

    lg_pop_scope(ctx.scratch, scope);
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

    LG_Arena scratch_allocator = {0};
    lg_arena_init(&scratch_allocator, &libc_allocator);

    (void)text;

    MRV_AST ast = mrv_parse(
        &libc_allocator,
        &libc_allocator,
        &libc_writer,
        &tstream,
        text
    );

    MRV_LanguageDescriptor ldesc = {0};
    mrv_analyze(
        &libc_allocator,
        &libc_allocator,
        &ast,
        text,
        &libc_writer,
        &ldesc
    );

    mrv_gen_source(&libc_writer, &scratch_allocator, &ldesc);

    mrv_ldesc_destroy(&ldesc);
    mrv_tstream_destroy(&tstream, &libc_allocator);
    mrv_ast_destroy(&ast);
    fclose(file);
    lg_arena_free_all(&scratch_allocator);
    return 0;
}

#include <libgrad/internal/base.c>
