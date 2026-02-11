# Grilly MCP Server

TypeScript MCP server that helps AI coders use [Grilly](https://github.com/grillcheese-ai/grilly) — the GPU-accelerated neural network framework using Vulkan.

## Tools

| Tool | Description |
|------|-------------|
| `grilly_docs` | Get API documentation (loads from `./docs` when available; topics: overview, quickstart, snn, fnn, attention, etc.) |
| `grilly_docs_file` | Read a specific doc file from `./docs` (e.g. `getting_started/quickstart.rst`) |
| `grilly_docs_list` | List available docs in `./docs` |
| `grilly_example` | Get copy-paste example code for an operation |
| `grilly_list_ops` | List available backend operations |
| `grilly_run_python` | Execute a Python snippet that uses Grilly (requires grilly + numpy installed) |

## Resources

| URI | Description |
|-----|-------------|
| `grilly://docs/overview` | API overview and backend namespaces |
| `grilly://docs/quickstart` | Quick start code examples |
| `grilly://docs/operations` | List of operations |

## Setup

### Install

```bash
cd mcp-servers/grilly
npm install
npm run build
```

### Cursor

Add to Cursor MCP settings (`.cursor/mcp.json` or Cursor Settings → MCP):

```json
{
  "mcpServers": {
    "grilly": {
      "command": "node",
      "args": ["c:/path/to/grilly/mcp-servers/grilly/dist/index.js"]
    }
  }
}
```

Use an absolute path to `dist/index.js`.

### Claude Code / Other MCP clients

```bash
# stdio transport
node mcp-servers/grilly/dist/index.js
```

Or configure your client to spawn the above command.

## Documentation

The server loads docs from `./docs` when available. Path resolution:

1. `GRILLY_DOCS_PATH` env var (absolute path)
2. `cwd/docs` (when run from repo root)
3. `../../docs` relative to the built server

If docs are missing, embedded fallback content is used.

## Testing

Run the quick test (from repo root: `cd ..; cd ..`):

```bash
node mcp-servers/grilly/test-mcp.mjs
# or: npm run test:mcp  (from mcp-servers/grilly)
```

This exercises `grilly_docs_list`, `grilly_docs`, and `grilly_docs_file` against `./docs`.

## Requirements

- Node.js >= 18
- For `grilly_run_python`: Python 3.12+, `grilly`, `numpy` installed

## License

MIT
