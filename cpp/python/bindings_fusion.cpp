/// bindings_fusion.cpp — JIT Shader Fusion Engine Python bindings.

#include "bindings_core.h"
#include "grilly/shader_fusion.h"

using namespace grilly;

void register_fusion_ops(py::module_& m) {

    py::class_<ShaderFusionEngine>(m, "ShaderFusionEngine")
        .def(py::init<const std::string&>(), py::arg("snippet_dir"),
             "Create fusion engine with snippet directory path.")
        .def("fuse",
            [](ShaderFusionEngine& engine, GrillyCoreContext& ctx,
               py::list op_list, uint32_t totalElements) -> py::dict {

                FusionGraph graph;
                graph.totalElements = totalElements;
                graph.numInputBuffers = 1;
                graph.numOutputBuffers = 1;

                for (auto& item : op_list) {
                    auto op_dict = item.cast<py::dict>();
                    FusionOp op;
                    op.opType = op_dict["type"].cast<std::string>();
                    op.numInputs = 1;
                    op.numOutputs = 1;

                    if (op_dict.contains("params")) {
                        auto params = op_dict["params"].cast<py::dict>();
                        for (auto& [k, v] : params) {
                            op.params[k.cast<std::string>()] = v.cast<float>();
                        }
                    }

                    if (op.opType == "add" || op.opType == "residual") {
                        graph.numInputBuffers = 2;
                    }
                    graph.ops.push_back(std::move(op));
                }

                FusionResult result = engine.fuse(graph, ctx.cache);

                py::dict d;
                d["shader_name"] = result.shaderName;
                d["cache_hit"] = result.cacheHit;
                d["compile_ms"] = result.compileTimeMs;
                if (!result.cacheHit) {
                    d["glsl_source"] = result.glslSource;
                }
                return d;
            },
            py::arg("device"), py::arg("ops"), py::arg("total_elements"),
            "Fuse a sequence of ops into a single compiled shader.\n"
            "ops: list of dicts with 'type' and optional 'params'.\n"
            "Returns dict with shader_name, cache_hit, compile_ms.")
        .def("stats",
            [](const ShaderFusionEngine& engine) -> py::dict {
                auto s = engine.stats();
                py::dict d;
                d["cache_hits"] = s.cacheHits;
                d["cache_misses"] = s.cacheMisses;
                d["total_compile_ms"] = s.totalCompileMs;
                d["shaders_compiled"] = s.shadersCompiled;
                return d;
            },
            "Get fusion engine statistics.");
}
