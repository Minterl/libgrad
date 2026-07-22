#ifndef LG_VM_H_
#define LG_VM_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

/// Discriminator for an operation.
///
/// The integer representations of opcodes are not designed
/// to be stable and should not be serialized.
enum lg_ir_opcode {
    /// --- Symbolic Unary Operations ---
#   define LG__FIRST_UNARY_OP LG_OPCODE_SOURCE
    LG_OPCODE_SOURCE,
    LG_OPCODE_SINK,

    /// --- Non-Symbolic Unary Operations ---
    LG_OPCODE_NOP,
    /// Element-wise ReLU
    LG_OPCODE_RELU,
    /// Element-wise stable softmax
    LG_OPCODE_STABLE_SOFTMAX,
    /// Element-wise sigmoid
    LG_OPCODE_SIGMOID,
    /// Element-wise natural log
    LG_OPCODE_LN,
#   define LG__LAST_UNARY_OP LG_OPCODE_LN

    /// --- Binary Operations ---
#   define LG__FIRST_BINARY_OP LG_OPCODE_ADD
    /// Element-wise tensor addition
    LG_OPCODE_ADD,
    /// Element-wise tensor subtraction
    LG_OPCODE_SUB,
    /// Generalized tensor contraction i.e
    /// dot-product over strided dimensions.
    /// Is generalizable to N-rank tensors.
    LG_OPCODE_CONTRACT,
    /// Hadamard product
    LG_OPCODE_HADAMARD,
    /// Mean Squared Error loss
    LG_OPCODE_LOSS_MSE,
    /// Cross-entropy loss
    LG_OPCODE_LOSS_CROSS_ENTROPY,
#   define LG__LAST_BINARY_OP LG_OPCODE_LOSS_CROSS_ENTROPY
};

_Static_assert(LG__LAST_UNARY_OP + 1 == LG__FIRST_BINARY_OP, "opcodes must be contigugous");

#define LG__OPCODE_IS_UNARY(op) ((LG__FIRST_UNARY_OP <= (op)) && ((op) <= LG__LAST_UNARY_OP))
#define LG__OPCODE_IS_BINARY(op) ((LG__FIRST_BINARY_OP <= (op)) && ((op) <= LG__LAST_BINARY_OP))

struct lg_ir_symbol {
    uint32_t  id;
};

struct lg_ir_expr_node_meta_contract {
    size_t n_contracted_axes;
    size_t n_batch_axes;
};

/// An IR node in an expr.
///
/// If x1.data == NULL, then the node represents a unary
/// operation.
/// Otherwise, it represents a binary operation as expected.
struct lg_ir_expr_node {
    enum lg_ir_opcode    opcode;

    struct lg_ir_symbol  y_logical;
    struct lg_desc       y_physical;
    uint32_t             y_buf_id;
    size_t               y_offset;

    struct lg_ir_symbol  x0_logical;
    struct lg_desc       x0_physical;
    uint32_t             x0_buf_id;
    size_t               x0_offset;

    struct lg_ir_symbol  x1_logical;
    struct lg_desc       x1_physical;
    uint32_t             x1_buf_id;
    size_t               x1_offset;

    union {
        struct lg_ir_expr_node_meta_contract contract;
    } meta_as;
};


/// The intermediate representation of a program.
/// 
/// As of right now, the exprs themselves do not support any control flow.
struct lg_ir_expr {
    size_t                   nodes_cap;
    size_t                   nodes_len;
    struct lg_ir_expr_node  *nodes LG_CHECK_BOUNDS(nodes_len);

    size_t                   buf_table_cap;
    size_t                   buf_table_len;
    uint32_t                *buf_table_ids LG_CHECK_BOUNDS(buf_table_len);
    size_t                  *buf_table_bytes_required LG_CHECK_BOUNDS(buf_table_len);

    uint32_t                 next_symbol_id;
};

enum lg_status LG_IR_DeclareSource(
    struct lg_ir_symbol *out_symbol,
    struct lg_desc physical_desc,
    struct lg_ir_expr *expr,
    uint32_t buf_id
);
enum lg_status LG_IR_DeclareSink(struct lg_ir_symbol sym, struct lg_ir_expr *expr);
enum lg_status LG_IR_GetSinkLocation(
    uint32_t *LG_NULLABLE out_buf_id,
    size_t *LG_NULLABLE out_offset,
    struct lg_desc *LG_NULLABLE out_desc,
    struct lg_ir_symbol sym,
    struct lg_ir_expr *expr
);

enum lg_status LG_IR_BuftabInsert(struct lg_ir_expr *expr, uint32_t id, size_t bytes_required);
enum lg_status LG_IR_BuftabGetIdx(size_t *LG_NULLABLE out_idx, const struct lg_ir_expr *expr, uint32_t id);

/// Gets the last physical location of the tensor `x` and populates
/// its `data` pointer if found.
///
/// This does not guarantee that the value will actually exist at the end
/// of execution. 
/// 
/// If you want to make sure that is the case, append a NOP using `lgvm_Nop` to 
/// the end of the expr.
enum lg_status LG_IR_AppendNop(struct lg_ir_expr *expr, struct lg_ir_symbol x);

enum lg_status LG_IR_AppendAdd(struct lg_ir_expr *expr, struct lg_ir_symbol *y, const struct lg_ir_symbol x0, const struct lg_ir_symbol x1);
enum lg_status LG_IR_AppendSub(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol x0, const struct lg_ir_symbol x1);
enum lg_status LG_IR_AppendContract(struct lg_ir_expr *expr, struct lg_ir_symbol *y, struct lg_ir_symbol x0, struct lg_ir_symbol x1, size_t n_contracted_axes, size_t n_batch_axes);
enum lg_status LG_IR_AppendHadamard(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol x0, const struct lg_ir_symbol x1);

enum lg_status LG_IR_AppendMSELoss(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol x0, const struct lg_ir_symbol x1);
enum lg_status LG_IR_AppendCrossEntropyLoss(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol x0, const struct lg_ir_symbol x1);

enum lg_status LG_IR_AppendReLU(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol in);
enum lg_status LG_IR_AppendStableSoftmax(struct lg_ir_expr *expr, const struct lg_ir_symbol y, const struct lg_ir_symbol in);
enum lg_status LG_IR_AppendSigmoid(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol in);
enum lg_status LG_IR_AppendLn(struct lg_ir_expr *expr, struct lg_ir_symbol y, const struct lg_ir_symbol in);

/// "Compiles" an expr.
enum lg_status LG_IR_CompileExpr(
    size_t *LG_NULLABLE out_bytes_required,
    struct lg_allocator *scratch,
    struct lg_ir_expr *expr,
    size_t mem_align
);

/// Allocate the memory necessary for an expr with the given capacities,
/// and assign offsets into the buffer for each field.
enum lg_status LG_AllocExpr(
    struct lg_allocator *perm,
    uint8_t *LG_NULLABLE *out_ptr,
    size_t *LG_NULLABLE out_bytes_allocated,
    struct lg_ir_expr *expr,
    size_t nodes_cap,
    size_t bufmap_cap
);

/// Frees the memory required for an expr.
void LG_FreeExpr(struct lg_allocator *allocator, struct lg_ir_expr *expr);

#endif // LG_VM_H_
