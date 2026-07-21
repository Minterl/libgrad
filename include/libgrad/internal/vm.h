#ifndef LG_VM_H_
#define LG_VM_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>

/// Discriminator for an operation.
///
/// The integer representations of opcodes are not designed
/// to be stable and should not be serialized.
enum lg_ir_opcode {
    /// --- BINARY OPERATIONS ---
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

    /// --- UNARY OPERATIONS ---
    LG_OPCODE_NOP,
    /// Element-wise ReLU
    LG_OPCODE_RELU,
    /// Element-wise stable softmax
    LG_OPCODE_STABLE_SOFTMAX,
    /// Element-wise sigmoid
    LG_OPCODE_SIGMOID,
    /// Element-wise natural log
    LG_OPCODE_LN,
};

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
    size_t                   cap;
    size_t                   len;
    
    struct lg_ir_expr_node  *nodes LG_CHECK_BOUNDS(len);

    size_t                   next_symbol_id;
};

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
    size_t *out_bytes_required,
    struct lg_allocator *scratch,
    struct lg_ir_expr *expr,
    size_t mem_align
);

/// Allocate the memory necessary for an expr of capacity `cap`,
/// and assign offsets into the buffer for each item in the SoA.
enum lg_status LG_AllocExpr(
    struct lg_allocator *allocator,
    uint8_t **out_ptr,
    size_t *out_bytes_allocated,
    struct lg_ir_expr *expr,
    size_t cap
);

/// Frees the memory required for an expr.
void LG_FreeExpr(struct lg_allocator *allocator, struct lg_ir_expr *expr);

#endif // LG_VM_H_
