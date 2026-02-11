#!/usr/bin/env node
/**
 * Quick test of the Grilly MCP server - sends JSON-RPC over stdio and captures responses.
 * Run: node mcp-servers/grilly/test-mcp.mjs  (from repo root)
 */
import { spawn } from "child_process";
import { createInterface } from "readline";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SERVER_PATH = join(__dirname, "dist", "index.js");
// Repo root = two levels up from mcp-servers/grilly
const REPO_ROOT = join(__dirname, "..", "..");

function send(msg) {
  const line = JSON.stringify(msg) + "\n";
  proc.stdin.write(line);
}

let proc;
const responses = [];

function run() {
  proc = spawn("node", [SERVER_PATH], {
    cwd: REPO_ROOT || process.cwd(),
    stdio: ["pipe", "pipe", "pipe"],
  });

  const rl = createInterface({ input: proc.stdout, crlfDelay: Infinity });
  rl.on("line", (line) => {
    try {
      const msg = JSON.parse(line);
      responses.push(msg);
      if (msg.id !== undefined && msg.result !== undefined) {
        console.log("Response:", JSON.stringify(msg, null, 2).slice(0, 500) + (JSON.stringify(msg).length > 500 ? "..." : ""));
      }
    } catch (e) {
      console.log("Raw:", line.slice(0, 200));
    }
  });

  proc.stderr.on("data", (d) => process.stderr.write(d));

  proc.on("close", (code) => {
    console.log("\nServer exited:", code);
    const listResult = responses.find((r) => r.id === 3 && r.result?.content);
    const docsResult = responses.find((r) => r.id === 4 && r.result?.content);
    const fileResult = responses.find((r) => r.id === 5 && r.result?.content);
    if (listResult?.result?.content?.[0]?.text) {
      console.log("\n--- grilly_docs_list ---\n");
      console.log(listResult.result.content[0].text);
    }
    if (docsResult?.result?.content?.[0]?.text) {
      console.log("\n--- grilly_docs quickstart (first 400 chars) ---\n");
      console.log(docsResult.result.content[0].text.slice(0, 400) + "...");
    }
    if (fileResult?.result?.content?.[0]?.text) {
      console.log("\n--- grilly_docs_file snn (first 300 chars) ---\n");
      console.log(fileResult.result.content[0].text.slice(0, 300) + "...");
    }
    process.exit(code ?? 0);
  });

  // 1. Initialize
  send({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "test", version: "0.1.0" },
    },
  });

  // 2. After init, send initialized notification and then tools/list and tools/call
  setTimeout(() => {
    send({ jsonrpc: "2.0", method: "notifications/initialized" });
    send({
      jsonrpc: "2.0",
      id: 2,
      method: "tools/list",
    });
  }, 100);

  setTimeout(() => {
    send({
      jsonrpc: "2.0",
      id: 3,
      method: "tools/call",
      params: { name: "grilly_docs_list", arguments: {} },
    });
  }, 200);

  setTimeout(() => {
    send({
      jsonrpc: "2.0",
      id: 4,
      method: "tools/call",
      params: { name: "grilly_docs", arguments: { topic: "quickstart" } },
    });
  }, 300);

  setTimeout(() => {
    send({
      jsonrpc: "2.0",
      id: 5,
      method: "tools/call",
      params: { name: "grilly_docs_file", arguments: { path: "concepts/spiking_neural_networks.rst" } },
    });
  }, 400);

  setTimeout(() => {
    proc.stdin.end();
  }, 600);
}

run();
