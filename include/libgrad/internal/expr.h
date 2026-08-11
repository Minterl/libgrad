#ifndef LG_EXPR_H_
#define LG_EXPR_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/vm.h> // TODO: maybe this should be the thing that defines logical opcodes and symbols


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// common expr types etc.
///
////////////////////////////////////////////////////////////////////////////////

// TODO: this should be moved to another file, maybe
typedef struct
LG_Context {
    LG_Arena         arena;

    LG_StatusKind    last_status;
    size_t           err_msg_len;
    uint8_t          err_msg_backing_buf[LG_MAX_ERR_LEN];
} LG_Context;

typedef uint32_t 
LG_SymbolFlags;
enum {
    LG_SymbolFlag_Pin = UINT32_C(0x1),
};

typedef union
LG_ExprNodeMeta {
    struct {
        size_t n_contracted_axes;
        size_t n_batch_axes;
    } contract;

    struct {
        LG_LogicalShape y_shape; 
    } param;
} LG_ExprNodeMeta;

typedef struct
LG_LogicalExprNode {
    LG_Opcode        opcode;

    LG_Symbol        y;
    LG_Symbol        x0;
    LG_Symbol        x1;

    LG_SymbolFlags   y_flags;
    LG_ExprNodeMeta  meta_as;
} LG_LogicalExprNode;

typedef struct 
LG_LogicalExpr {
    size_t max_symbol_id;
    size_t len;
    LG_LogicalExprNode *nodes lg_check_bounds(len);
} LG_LogicalExpr;


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// expr builder
///
////////////////////////////////////////////////////////////////////////////////

typedef struct
LG_BuilderNode {
    /// Nodes are stored in reverse-chronological order, so we only have a
    /// prev pointer
    struct LG_BuilderNode *prev;

    LG_Opcode opcode;

    LG_Symbol x0;
    LG_Symbol x1;
    LG_Symbol y;

    LG_SymbolFlags   y_flags;
    LG_ExprNodeMeta  meta_as;
} LG_BuilderNode;

typedef struct
LG_Builder {
    LG_BuilderNode  *ir_tail;
    uint32_t         next_symbol_id;
} LG_Builder;

LG_Symbol
lg_param(LG_Context *ctx, LG_Builder *lexpr, LG_LogicalShape shape);

void
lg_pin(LG_Context *ctx, LG_Builder *lexpr, LG_Symbol sym);

void
lg_force_layout(LG_Context *ctx, LG_Builder *lexpr, LG_Symbol sym, LG_LayoutKind layout);

LG_Symbol
lg_add(LG_Context *ctx, LG_Builder *lexpr, LG_Symbol x0, LG_Symbol x1);

LG_Symbol
lg_contract(
    LG_Context *ctx,
    LG_Builder *lexpr,
    LG_Symbol x0,
    LG_Symbol x1,
    size_t n_contracted_axes,
    size_t n_batch_axes
);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///
/// physical expr stuff
///
////////////////////////////////////////////////////////////////////////////////

typedef uint32_t LG_LogicalExprLoweringFlags;
enum {
    LG_LogicalExprLoweringFlag_NoStructuralInvariantValidation = (0x1),
};

typedef struct
LG_PhysicalExprNode {
    LG_Opcode            opcode;

    LG_StridedDesc       y_physical;
    uint32_t             y_buf_id;
    size_t               y_offset;

    LG_StridedDesc       x0_physical;
    uint32_t             x0_buf_id;
    size_t               x0_offset;

    LG_StridedDesc       x1_physical;
    uint32_t             x1_buf_id;
    size_t               x1_offset;

    LG_ExprNodeMeta      meta_as;
} LG_PhysicalExprNode;

typedef struct
LG_PhysicalExpr {
    size_t                len;
    LG_PhysicalExprNode  *nodes lg_check_bounds(len);
} LG_PhysicalExpr;

#endif // LG_EXPR_H_
