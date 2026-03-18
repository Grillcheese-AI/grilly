/// bindings_optim.cpp — Standalone optimizer step op bindings.
///
/// Stub: will be populated with adam_step, sgd_step, etc.
/// (the raw GPU dispatch ops, not the nn::Optimizer classes which
/// are in bindings_core.cpp) from the monolithic bindings.cpp.

#include "bindings_core.h"

void register_optim_ops(py::module_& m) {
    // TODO: Move standalone optimizer step ops from bindings.cpp
    (void)m;
}
