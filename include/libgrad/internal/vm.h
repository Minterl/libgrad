#ifndef LG_VM_H_
#define LG_VM_H_

#include <libgrad/internal/core.h>

/// Represents a single tensor backed by `data`
///
/// Generally, these are thin handles that should 
/// live on the stack and be shallow copied as necessary.
///
/// The size `data`, respectively, should product of the
/// non-zero values in `dim`.
///
/// Backing buffers are stored stride-major.
struct lg_ir_tensor {
    /// Since the exprs are pure SSA, the time the (y) value is born at
    /// functions as a globally unique identifier.
    uint32_t born_at;
    
    /// The shape descriptor of the tensor.
    struct lg_desc desc;
    
    /// The tensor's primary backing buffer.
    lg_scalar *data;
};

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

/// An IR node in an expr.
///
/// If x1.data == NULL, then the node represents a unary
/// operation.
/// Otherwise, it represents a binary operation as expected.
struct lg_ir_expr_node {
    enum lg_ir_opcode    opcode;

    struct lg_ir_tensor  y;
    struct lg_ir_tensor  x0;
    struct lg_ir_tensor  x1;
};

/// The intermediate representation of a program.
/// 
/// As of right now, the exprs themselves do not support any control flow.
struct lg_ir_expr {
    size_t                   cap;
    size_t                   len;
    
    struct lg_ir_expr_node  *nodes LG_CHECK_BOUNDS(len);
};

/// Gets the last physical location of the tensor `x` and populates
/// its `data` pointer if found.
///
/// This does not guarantee that the value will actually exist at the end
/// of execution. 
/// 
/// If you want to make sure that is the case, append a NOP using `lgvm_Nop` to 
/// the end of the expr.
enum lg_status LG_IR_GetLastLocation(struct lg_ir_expr *expr, struct lg_ir_tensor *x);

enum lg_status LG_IR_AppendNop(struct lg_ir_expr *expr, struct lg_ir_tensor x);

enum lg_status LG_IR_AppendAdd(struct lg_ir_expr *expr, struct lg_ir_tensor *y, const struct lg_ir_tensor x0, const struct lg_ir_tensor x1);
enum lg_status LG_IR_AppendSub(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor x0, const struct lg_ir_tensor x1);
enum lg_status LG_IR_AppendContract(struct lg_ir_expr *expr, struct lg_ir_tensor *y, struct lg_ir_tensor x0, struct lg_ir_tensor x1, size_t n_contracted_axes, size_t n_batch_axes);
enum lg_status LG_IR_AppendHadamard(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor x0, const struct lg_ir_tensor x1);

enum lg_status LG_IR_AppendMSELoss(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor x0, const struct lg_ir_tensor x1);
enum lg_status LG_IR_AppendCrossEntropyLoss(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor x0, const struct lg_ir_tensor x1);

enum lg_status LG_IR_AppendReLU(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor in);
enum lg_status LG_IR_AppendStableSoftmax(struct lg_ir_expr *expr, const struct lg_ir_tensor y, const struct lg_ir_tensor in);
enum lg_status LG_IR_AppendSigmoid(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor in);
enum lg_status LG_IR_AppendLn(struct lg_ir_expr *expr, struct lg_ir_tensor y, const struct lg_ir_tensor in);

/// "Compiles" an expr.
enum lg_status LG_IR_CompileExpr(struct lg_ir_expr *expr);

#endif // LG_VM_H_
