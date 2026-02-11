#!/usr/bin/env node

/**
 * Grilly MCP Server
 *
 * Helps AI coders use Grilly (GPU-accelerated neural network framework).
 * Loads documentation from ./docs (RST), with embedded fallback when unavailable.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { spawn } from "child_process";
import { readFileSync, existsSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Resolve docs path: GRILLY_DOCS_PATH env, or repo root docs/, or ../../docs from dist/
function getDocsPath(): string | null {
  const env = process.env.GRILLY_DOCS_PATH;
  if (env) return env;
  const cwdDocs = join(process.cwd(), "docs");
  if (existsSync(cwdDocs)) return cwdDocs;
  const relDocs = join(__dirname, "..", "..", "docs");
  if (existsSync(relDocs)) return relDocs;
  return null;
}

// Topic -> docs path (relative to docs/)
const TOPIC_TO_PATH: Record<string, string> = {
  overview: "concepts/architecture_and_runtime.rst",
  quickstart: "getting_started/quickstart.rst",
  installation: "getting_started/installation.rst",
  core_concepts: "getting_started/core_concepts.rst",
  pytorch_users: "getting_started/pytorch_users.rst",
  snn: "concepts/spiking_neural_networks.rst",
  fnn: "concepts/module_system_and_functional_api.rst",
  attention: "concepts/attention_transformers_and_decoding.rst",
  memory: "concepts/memory_retrieval_and_faiss.rst",
  faiss: "concepts/memory_retrieval_and_faiss.rst",
  lora: "concepts/lora_and_adapter_finetuning.rst",
  training: "concepts/training_and_optimization.rst",
  convolution: "concepts/convolution_pooling_and_normalization.rst",
  design: "concepts/design_choices.rst",
  tensor: "concepts/tensor_model_and_shapes.rst",
  multimodal: "concepts/multimodal_capsule_and_vlm.rst",
  performance: "concepts/performance_debugging_and_testing.rst",
};

function readDocFile(docsRoot: string, relPath: string): string | null {
  const full = join(docsRoot, relPath);
  if (!existsSync(full)) return null;
  try {
    return readFileSync(full, "utf-8");
  } catch {
    return null;
  }
}

function listDocFiles(docsRoot: string): string[] {
  const out: string[] = [];
  function walk(dir: string, prefix: string) {
    const base = dir ? join(docsRoot, dir) : docsRoot;
    if (!existsSync(base)) return;
    const entries = readdirSync(base, { withFileTypes: true });
    for (const e of entries) {
      const rel = prefix ? `${prefix}/${e.name}` : e.name;
      if (e.isDirectory() && !e.name.startsWith("_")) {
        walk(dir ? `${dir}/${e.name}` : e.name, rel);
      } else if (e.isFile() && (e.name.endsWith(".rst") || e.name.endsWith(".md"))) {
        out.push(rel);
      }
    }
  }
  walk("", "");
  return out.sort();
}

// ---------------------------------------------------------------------------
// Embedded fallback (when docs/ unavailable)
// ---------------------------------------------------------------------------

const GRILLY_OVERVIEW = `# Grilly - GPU-Accelerated Neural Network Framework

PyTorch-like API using Vulkan compute shaders. Works on AMD, NVIDIA, Intel GPUs — no CUDA.

## Entry Point
\`\`\`python
import grilly
backend = grilly.Compute()
\`\`\`

## Backend Namespaces
- \`backend.snn.*\` — Spiking neural networks (LIF, GIF, STDP, Hebbian)
- \`backend.fnn.*\` — Feedforward (linear, activations, layernorm)
- \`backend.attention.*\` — Flash Attention 2, multi-head attention
- \`backend.memory.*\` — Memory read/write, context aggregation
- \`backend.faiss.*\` — Vector similarity (distances, topk, IVF)
- \`backend.learning.*\` — Adam, EWC, NLMS, natural gradients
- \`backend.cells.*\` — Place and time cells

## Data Format
All data is \`np.float32\` numpy arrays. Backend handles GPU upload/download.`;

const GRILLY_QUICKSTART = `# Grilly Quick Start

\`\`\`python
import grilly
import numpy as np

backend = grilly.Compute()

# SNN - LIF step
input_current = np.random.randn(1000).astype(np.float32)
membrane = np.zeros(1000, dtype=np.float32)
refractory = np.zeros(1000, dtype=np.float32)
membrane, refractory, spikes = backend.snn.lif_step(
    input_current, membrane, refractory,
    dt=0.001, tau_mem=20.0, v_thresh=1.0
)

# FNN - Linear + SwiGLU
x = np.random.randn(32, 384).astype(np.float32)
weight = np.random.randn(384, 128).astype(np.float32)
bias = np.zeros(128, dtype=np.float32)
output = backend.fnn.linear(x, weight, bias)
activated = backend.fnn.swiglu(output)

# Flash Attention 2
q = np.random.randn(32, 8, 64, 64).astype(np.float32)  # (batch, heads, seq, dim)
k = np.random.randn(32, 8, 64, 64).astype(np.float32)
v = np.random.randn(32, 8, 64, 64).astype(np.float32)
attention_out = backend.attention.flash_attention2(q, k, v)

# FAISS similarity search
query = np.random.randn(1, 384).astype(np.float32)
database = np.random.randn(10000, 384).astype(np.float32)
distances = backend.faiss.compute_distances(query, database)
top_k_dist, top_k_idx = backend.faiss.topk(distances, k=10)
\`\`\`

## nn Module (PyTorch-like)
\`\`\`python
from grilly.nn import Linear, LayerNorm, Parameter
from grilly.optim import Adam
\`\`\`

## Requirements
- Python >= 3.12
- Vulkan drivers
- NumPy`;

const EXAMPLES: Record<string, string> = {
  snn: `# SNN - Spiking Neural Network
\`\`\`python
import grilly
import numpy as np

backend = grilly.Compute()
# LIF neuron step
input_current = np.random.randn(1000).astype(np.float32)
membrane = np.zeros(1000, dtype=np.float32)
refractory = np.zeros(1000, dtype=np.float32)
membrane, refractory, spikes = backend.snn.lif_step(
    input_current, membrane, refractory,
    dt=0.001, tau_mem=20.0, v_thresh=1.0
)
\`\`\``,
  fnn: `# FNN - Feedforward (Linear, Activations)
\`\`\`python
import grilly
import numpy as np

backend = grilly.Compute()
x = np.random.randn(32, 384).astype(np.float32)
w = np.random.randn(384, 128).astype(np.float32)
b = np.zeros(128, dtype=np.float32)
out = backend.fnn.linear(x, w, b)
activated = backend.fnn.swiglu(out)  # or activation_relu, activation_gelu
\`\`\``,
  attention: `# Flash Attention 2
\`\`\`python
import grilly
import numpy as np

backend = grilly.Compute()
# (batch, heads, seq, dim)
q = np.random.randn(32, 8, 64, 64).astype(np.float32)
k = np.random.randn(32, 8, 64, 64).astype(np.float32)
v = np.random.randn(32, 8, 64, 64).astype(np.float32)
out = backend.attention.flash_attention2(q, k, v)
\`\`\``,
  faiss: `# FAISS - Vector Similarity Search
\`\`\`python
import grilly
import numpy as np

backend = grilly.Compute()
query = np.random.randn(1, 384).astype(np.float32)
database = np.random.randn(10000, 384).astype(np.float32)
distances = backend.faiss.compute_distances(query, database)
top_k_dist, top_k_idx = backend.faiss.topk(distances, k=10)
\`\`\``,
  nn_module: `# nn Module - PyTorch-like Layers
\`\`\`python
from grilly.nn import Linear, LayerNorm, Parameter
from grilly.optim import Adam
import numpy as np

model = Linear(384, 128)
x = np.random.randn(4, 384).astype(np.float32)
out = model(x)
# Train with Adam
opt = Adam(model.parameters(), lr=1e-3)
\`\`\``,
};

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "grilly",
  version: "0.1.0",
});

// grilly_docs - Get API documentation (from ./docs when available)
server.registerTool(
  "grilly_docs",
  {
    title: "Grilly API Docs",
    description:
      "Get API documentation for Grilly. Uses ./docs when available. Topics: overview, quickstart, snn, fnn, attention, faiss, nn_module, installation, core_concepts, training, lora, memory, convolution, design, tensor, multimodal, performance.",
    inputSchema: {
      topic: z
        .string()
        .describe("Topic: overview, quickstart, snn, fnn, attention, faiss, nn_module, etc."),
    },
  },
  async (args) => {
    const topic = (args.topic ?? "overview").toLowerCase();
    const docsRoot = getDocsPath();
    if (docsRoot && topic in TOPIC_TO_PATH) {
      const raw = readDocFile(docsRoot, TOPIC_TO_PATH[topic]);
      if (raw) {
        return { content: [{ type: "text" as const, text: raw }] };
      }
    }
    // Fallback to embedded
    let content: string;
    if (topic === "overview") {
      content = GRILLY_OVERVIEW;
    } else if (topic === "quickstart") {
      content = GRILLY_QUICKSTART;
    } else if (topic in EXAMPLES) {
      content = EXAMPLES[topic] ?? `Unknown topic: ${topic}`;
    } else {
      const topics = Object.keys(TOPIC_TO_PATH).join(", ");
      content = `Available topics: ${topics}\n\nGot: ${topic}`;
    }
    return { content: [{ type: "text" as const, text: content }] };
  }
);

// grilly_docs_file - Read a specific doc file from ./docs
server.registerTool(
  "grilly_docs_file",
  {
    title: "Read Grilly Doc File",
    description:
      "Read a specific documentation file from ./docs. Use a path like 'getting_started/quickstart.rst', 'concepts/spiking_neural_networks.rst', or 'api/grilly.functional.rst'. Use grilly_docs_list to see available files.",
    inputSchema: {
      path: z.string().describe("Relative path within docs/, e.g. getting_started/quickstart.rst"),
    },
  },
  async (args) => {
    const docsRoot = getDocsPath();
    if (!docsRoot) {
      return { content: [{ type: "text" as const, text: "Docs folder not found. Set GRILLY_DOCS_PATH or run from repo root." }] };
    }
    const rel = args.path.replace(/^\/+/, "").replace(/\\/g, "/");
    const raw = readDocFile(docsRoot, rel);
    if (!raw) {
      return { content: [{ type: "text" as const, text: `File not found: ${rel}. Use grilly_docs_list to see available files.` }] };
    }
    return { content: [{ type: "text" as const, text: raw }] };
  }
);

// grilly_docs_list - List available doc files in ./docs
server.registerTool(
  "grilly_docs_list",
  {
    title: "List Grilly Doc Files",
    description: "List available documentation files in ./docs (getting_started, concepts, tutorials, api).",
    inputSchema: {},
  },
  async () => {
    const docsRoot = getDocsPath();
    if (!docsRoot) {
      return { content: [{ type: "text" as const, text: "Docs folder not found. Set GRILLY_DOCS_PATH or run from repo root." }] };
    }
    const files = listDocFiles(docsRoot);
    const text = files.length ? files.join("\n") : "No .rst or .md files found in docs.";
    return { content: [{ type: "text" as const, text }] };
  }
);

// grilly_example - Get example code for an operation
server.registerTool(
  "grilly_example",
  {
    title: "Grilly Example Code",
    description:
      "Get copy-paste example code for a Grilly operation. Options: snn, fnn, attention, faiss, nn_module.",
    inputSchema: {
      operation: z
        .string()
        .describe("Operation: snn, fnn, attention, faiss, nn_module"),
    },
  },
  async (args) => {
    const op = (args.operation ?? "fnn").toLowerCase();
    const content = EXAMPLES[op] ?? `Unknown operation: ${op}. Available: ${Object.keys(EXAMPLES).join(", ")}`;
    return { content: [{ type: "text" as const, text: content }] };
  }
);

// grilly_list_ops - List available backend operations
server.registerTool(
  "grilly_list_ops",
  {
    title: "List Grilly Operations",
    description: "List available backend operation namespaces and key methods.",
    inputSchema: {},
  },
  async () => {
    const content = `# Grilly Backend Operations

## backend.snn
- lif_step(input_current, membrane, refractory, dt, tau_mem, v_thresh)
- gif_neuron_step, hebbian_learning, stdp_learning

## backend.fnn
- linear(x, weight, bias)
- activation_relu, activation_gelu, activation_swiglu
- layernorm, dropout

## backend.attention
- flash_attention2(q, k, v)

## backend.faiss
- compute_distances(query, database)
- topk(distances, k)

## backend.memory
- memory_read, memory_write

## backend.learning
- adam_update, ewc_penalty

## nn module
- Linear, LayerNorm, Embedding, Softmax
- Parameter, Module

## optim
- Adam, AdamW, SGD, NLMS`;
    return { content: [{ type: "text" as const, text: content }] };
  }
);

// grilly_run_python - Execute Python code that uses Grilly
server.registerTool(
  "grilly_run_python",
  {
    title: "Run Grilly Python Code",
    description:
      "Execute a Python code snippet that uses Grilly. Runs in a subprocess. Use for quick validation. Requires grilly and numpy installed.",
    inputSchema: {
      code: z.string().describe("Python code to execute (must import grilly/numpy)"),
    },
  },
  async (args) => {
    const { code } = args;
    return new Promise((resolve, reject) => {
      const proc = spawn("python", ["-c", code], {
        stdio: ["pipe", "pipe", "pipe"],
      });
      let stdout = "";
      let stderr = "";
      proc.stdout?.on("data", (d) => (stdout += d.toString()));
      proc.stderr?.on("data", (d) => (stderr += d.toString()));
      proc.on("close", (code) => {
        const out = stdout.trim() || "(no output)";
        const err = stderr.trim() ? `\n--- stderr ---\n${stderr}` : "";
        const text = `Exit code: ${code}\n${out}${err}`;
        resolve({ content: [{ type: "text" as const, text }] });
      });
      proc.on("error", (err) => {
        resolve({
          content: [{ type: "text" as const, text: `Failed to run: ${err.message}` }],
        });
      });
    });
  }
);

// ---------------------------------------------------------------------------
// Resources (from ./docs when available)
// ---------------------------------------------------------------------------

function getDocContent(topic: string, fallback: string): string {
  const docsRoot = getDocsPath();
  const docPath = TOPIC_TO_PATH[topic];
  if (docsRoot && docPath) {
    const raw = readDocFile(docsRoot, docPath);
    if (raw) return raw;
  }
  return fallback;
}

server.registerResource(
  "grilly-overview",
  "grilly://docs/overview",
  { title: "Grilly API Overview", description: "Architecture and backend namespaces" },
  async (uri) => ({
    contents: [{ uri: uri.toString(), text: getDocContent("overview", GRILLY_OVERVIEW) }],
  })
);

server.registerResource(
  "grilly-quickstart",
  "grilly://docs/quickstart",
  { title: "Grilly Quick Start", description: "Code examples for common operations" },
  async (uri) => ({
    contents: [{ uri: uri.toString(), text: getDocContent("quickstart", GRILLY_QUICKSTART) }],
  })
);

server.registerResource(
  "grilly-operations",
  "grilly://docs/operations",
  { title: "Grilly Operations", description: "List of backend operations" },
  async (uri) => ({
    contents: [
      {
        uri: uri.toString(),
        text: "backend.snn (lif_step, gif_neuron_step) | backend.fnn (linear, activations) | backend.attention (flash_attention2) | backend.faiss (compute_distances, topk) | backend.memory | backend.learning | nn (Linear, LayerNorm) | optim (Adam, SGD)",
      },
    ],
  })
);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error("Grilly MCP server error:", err);
  process.exit(1);
});
