#ifndef LG_VM_H_
#define LG_VM_H_

#include <libgrad/internal/core.h>
#include <libgrad/internal/alloc.h>
#include <libgrad/internal/vm_symtab.h>
#include <libgrad/internal/strings.h>
#include <libgrad/internal/map.h>

#define LG_MAX_ERR_LEN 1024

/// Discriminator for an operation.
///
/// The integer representations of opcodes are not designed
/// to be stable and should not be serialized.
typedef enum
LG_Opcode {

    //////////////////////////////////
    // Unary Operations
    //////////////////////////////////

#   define LG_FIRST_UNARY_OP LG_Opcode_Sink
    LG_Opcode_Sink,

    // Constructive operations create new symbols,
    // while non-constructive ones do not.
#   define LG_FIRST_CONSTRUCTIVE_OP LG_Opcode_Source

    LG_Opcode_Source,
    /// Element-wise ReLU
    LG_Opcode_ReLU,
    /// Element-wise stable softmax
    LG_Opcode_StableSoftmax,
    /// Element-wise sigmoid
    LG_Opcode_Sigmoid,
    /// Element-wise natural log
    LG_Opcode_LN,

#   define LG_LAST_UNARY_OP LG_Opcode_LN


    //////////////////////////////////
    // Binary Operations
    //////////////////////////////////

#   define LG_FIRST_BINARY_OP LG_Opcode_Add
    
    /// Element-wise tensor addition
    LG_Opcode_Add,
    /// Element-wise tensor subtraction
    LG_Opcode_Sub,
    /// Generalized tensor contraction i.e
    /// dot-product over strided dimensions.
    /// Is generalizable to N-rank tensors.
    LG_Opcode_Contract,
    /// Hadamard product
    LG_Opcode_Hadamard,
    /// Mean Squared Error loss
    LG_Opcode_MSELoss,
    /// Cross-entropy loss
    LG_Opcode_CrossEntropyLoss,

#   define LG_LAST_BINARY_OP LG_Opcode_CrossEntropyLoss
#   define LG_LAST_CONSTRUCTIVE_OP LG_LAST_BINARY_OP
} LG_Opcode;

_Static_assert(LG_LAST_UNARY_OP + 1 == LG_FIRST_BINARY_OP, "opcodes must be contigugous");

#define lg_opcode_creates_symbol(op) ((LG_FIRST_CONSTRUCTIVE_OP <= (op)) && ((op) <= LG_LAST_CONSTRUCTIVE_OP))
#define lg_opcode_is_unary(op) ((LG_FIRST_UNARY_OP <= (op)) && ((op) <= LG_LAST_UNARY_OP))
#define lg_opcode_is_binary(op) ((LG_FIRST_BINARY_OP <= (op)) && ((op) <= LG_LAST_BINARY_OP))

typedef struct
LG_Symbol {
    uint32_t id;
} LG_Symbol;

typedef union 
LG_ExprNodeMeta {
    struct {
        size_t n_contracted_axes;
        size_t n_batch_axes;
    } contract;
} LG_ExprNodeMeta;

/// An IR node in an expr.
typedef struct
LG_ExprNode {
    LG_Opcode            opcode;

    LG_Symbol            y_logical;
    LG_StridedDesc       y_physical;
    uint32_t             y_buf_id;
    size_t               y_offset;

    LG_Symbol            x0_logical;
    LG_StridedDesc       x0_physical;
    uint32_t             x0_buf_id;
    size_t               x0_offset;

    LG_Symbol            x1_logical;
    LG_StridedDesc       x1_physical;
    uint32_t             x1_buf_id;
    size_t               x1_offset;

    LG_ExprNodeMeta meta_as;
} LG_ExprNode;

typedef struct
LG_BufferTableEntry {
    size_t size_in_bytes;
} LG_BufferTableEntry;

typedef struct
LG_BufferTable {
    LG_Map map;
    LG_BufferTableEntry *entries;
} LG_BufferTable;

/// The intermediate representation of a program.
/// 
/// As of right now, the exprs themselves do not support any control flow.
typedef struct
LG_Expr {
    size_t          nodes_cap;
    size_t          nodes_len;
    LG_ExprNode    *nodes lg_check_bounds(nodes_len);

    LG_BufferTable  buftab;
} LG_Expr;

typedef struct
LG_CompilationContext {
    LG_Expr              *expr;
    LG_Allocator         *scratch;
    LG_SymbolTable        symtab;

    LG_StatusKind         last_status;
    size_t                err_msg_len;
    uint8_t               err_msg_backing_buf[LG_MAX_ERR_LEN];

    uint32_t              next_symbol_id;
} LG_CompilationContext;

void
lg_print_compilation_error(LG_CompilationContext *ctx, LG_Writer *writer);

LG_Symbol
lg_declare_source(
    LG_CompilationContext *ctx,
    LG_StridedDesc physical_desc,
    uint32_t buf_id
);

void
lg_declare_sink(LG_CompilationContext *ctx, LG_Symbol sym);

LG_StatusKind 
lg_get_sink_location(
    uint32_t *lg_nullable out_buf_id,
    size_t *lg_nullable out_offset,
    LG_StridedDesc *lg_nullable out_desc,
    LG_Symbol sym,
    LG_Expr *expr
);

LG_StatusKind 
lg_buftab_insert(LG_BufferTable *buftab, uint32_t id);

LG_StatusKind 
lg_buftab_get(LG_BufferTable *buftab, LG_BufferTableEntry *lg_nullable out_entry, uint32_t id);

LG_StatusKind
lg_buftab_update(LG_BufferTable *buftab, uint32_t id, LG_BufferTableEntry new_entry);

/// Gets the last physical location of the tensor `x` and populates
/// its `data` pointer if found.
///
/// This does not guarantee that the value will actually exist at the end
/// of execution. 
/// 
/// If you want to make sure that is the case, append a NOP using `lgvm_Nop` to 
/// the end of the expr.
LG_StatusKind 
lg_append_nop(LG_Expr *expr, LG_Symbol x);

LG_Symbol 
lg_append_add(LG_CompilationContext *ctx, const LG_Symbol x0, const LG_Symbol x1);
// LG_StatusKind 
// lg_append_sub(LG_Expr *expr, LG_Symbol y, const LG_Symbol x0, const LG_Symbol x1);
LG_Symbol 
lg_append_contract(
    LG_CompilationContext *ctx,
    LG_Symbol x0,
    LG_Symbol x1,
    size_t n_contracted_axes, 
    size_t n_batch_axes
);
// LG_StatusKind 
// lg_append_hadamard(LG_Expr *expr, LG_Symbol y, const LG_Symbol x0, const LG_Symbol x1);

// LG_StatusKind 
// lg_append_mse_loss(LG_Expr *expr, LG_Symbol y, const LG_Symbol x0, const LG_Symbol x1);
// LG_StatusKind 
// lg_append_cross_entropy_loss(LG_Expr *expr, LG_Symbol y, const LG_Symbol x0, const LG_Symbol x1);

// LG_StatusKind 
// lg_append_relu(LG_Expr *expr, LG_Symbol y, const LG_Symbol in);
// LG_StatusKind 
// lg_append_stable_softmax(LG_Expr *expr, const LG_Symbol y, const LG_Symbol in);
// LG_StatusKind 
// lg_append_sigmoid(LG_Expr *expr, LG_Symbol y, const LG_Symbol in);
// LG_StatusKind 
// lg_append_ln(LG_Expr *expr, LG_Symbol y, const LG_Symbol in);

/// "Compiles" an expr.
LG_StatusKind 
lg_compile_expr(
    LG_CompilationContext *ctx,
    size_t mem_align
);

/// Allocate the memory necessary for an expr with the given capacities,
/// and assign offsets into the buffer for each field.
LG_StatusKind 
lg_alloc_expr(
    LG_Allocator *alloc,
    LG_Expr *expr,
    size_t nodes_cap,
    size_t bufmap_cap
);

/// Frees the memory required for an expr.
void 
lg_free_expr(LG_Allocator *allocator, LG_Expr *expr);

#endif // LG_VM_H_
