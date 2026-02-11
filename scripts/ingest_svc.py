#!/usr/bin/env python3
"""SVC ingestion (fast, streaming).

The original ingestion flow was doing two full passes:
  1) InstantLanguage.ingest_svc(...)
  2) CognitiveController.ingest_svc(...) (which calls language.ingest_svc again)

That doubles the encoding work and makes large JSONL files feel like they
"hang". This script ingests **once** through CognitiveController, streaming
the JSONL in configurable chunks.

Usage:
  python scripts/ingest_svc.py -f datasets/_data/svc_training_merged.jsonl
  python scripts/ingest_svc.py -f ... --max 50000 --chunk 4096 --no-templates
  python scripts/ingest_svc.py -f ... --no-ngrams   # much faster vocab build
"""

import argparse
import sys
import time
from pathlib import Path


def _fmt_rate(n: int, dt: float) -> str:
    """Run fmt rate."""

    if dt <= 0:
        return "∞/s"
    rate = n / dt
    if rate >= 1e6:
        return f"{rate / 1e6:.2f}M/s"
    if rate >= 1e3:
        return f"{rate / 1e3:.2f}k/s"
    return f"{rate:.2f}/s"


def main() -> None:
    """Run main."""

    ap = argparse.ArgumentParser(description="Stream-ingest SVC JSONL into Grilly.")
    ap.add_argument("--file", "-f", required=True, help="Path to the SVC JSONL file")
    ap.add_argument("--max", "-n", type=int, default=None, help="Max entries to ingest")
    ap.add_argument("--realms", "-r", nargs="*", default=None, help="Only these realms")
    ap.add_argument("--min-complexity", type=float, default=None)
    ap.add_argument("--max-complexity", type=float, default=None)
    ap.add_argument("--sources", "-s", nargs="*", default=None, help="Only these sources")
    ap.add_argument("--dim", "-d", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=4096, help="Entries per ingestion chunk")
    ap.add_argument("--progress", type=int, default=50, help="Print progress every N entries")
    ap.add_argument("--no-templates", action="store_true", help="Skip template learning")
    ap.add_argument("--no-realm-vectors", action="store_true", help="Skip realm vector building")
    ap.add_argument(
        "--no-ngrams",
        action="store_true",
        help="Disable n-gram HRR word encoding (much faster, less lexical similarity)",
    )
    ap.add_argument("--verbose", "-v", action="store_true", default=False, help="Verbose output")
    args = ap.parse_args()

    # Imports here so --help is instant
    # Repo-layout compatibility: some setups package everything under "grilly",
    # others run directly from the repo root.
    try:
        from grilly.experimental.cognitive.controller import CognitiveController
        from grilly.experimental.language.svc_loader import SVCIngestionEngine, load_svc_entries
        from grilly.experimental.moe.routing import ResonatorMoE
        from grilly.experimental.vsa.ops import BinaryOps
        from grilly.utils.ingest_checkpoint import (
            CheckpointView,
            load_ingest_checkpoint,
            save_ingest_checkpoint,
        )
    except ModuleNotFoundError:
        from experimental.cognitive.controller import CognitiveController
        from experimental.language.svc_loader import SVCIngestionEngine, load_svc_entries
        from experimental.moe.routing import ResonatorMoE
        from experimental.vsa.ops import BinaryOps
        from utils.ingest_checkpoint import (
            save_ingest_checkpoint,
        )

    print("=" * 60)
    print("Grilly SVC Ingestion (streaming)")
    print("=" * 60)

    engine = SVCIngestionEngine(dim=args.dim)
    print(f"\nEngine: {engine.status()}")

    controller = CognitiveController(
        dim=args.dim,
        word_use_ngrams=not args.no_ngrams,
    )

    # Stream entries and ingest in chunks
    t0 = time.time()
    chunk = []
    total = 0
    last_print = t0
    total_templates = 0
    total_sentences = 0
    total_new_words = 0

    it = load_svc_entries(
        path=args.file,
        max_entries=args.max,
        realms=args.realms,
        min_complexity=args.min_complexity,
        max_complexity=args.max_complexity,
        sources=args.sources,
    )

    for entry in it:
        chunk.append(entry)
        if len(chunk) < args.chunk:
            continue

        res = controller.ingest_svc(
            chunk,
            learn_templates=not args.no_templates,
            build_realm_vectors=not args.no_realm_vectors,
            verbose=args.verbose,
            engine=engine,
        )
        total += len(chunk)
        total_templates += res.templates_learned
        total_sentences += res.sentences_learned
        total_new_words += res.words_encoded
        chunk.clear()

        if args.progress and total % args.progress == 0:
            now = time.time()
            print(
                f"  ... {total} ingested ({_fmt_rate(args.progress, now - last_print)}), "
                f"facts={len(controller.world.facts)}"
            )
            last_print = now

    # Final partial chunk
    if chunk:
        res = controller.ingest_svc(
            chunk,
            learn_templates=not args.no_templates,
            build_realm_vectors=not args.no_realm_vectors,
            verbose=args.verbose,
            engine=engine,
        )

        total += len(chunk)
        total_templates += res.templates_learned
        total_sentences += res.sentences_learned
        total_new_words += res.words_encoded

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {dt:.2f}s ({_fmt_rate(total, dt)})")
    print(f"  Entries:    {total}")
    print(f"  Sentences:  {total_sentences}")
    print(f"  New words:  {total_new_words}")
    print(f"  Facts:      {len(controller.world.facts)}")
    out = Path(args.file).with_suffix(".ingest_checkpoint")
    print(f"  Checkpoint: {out}")
    save_ingest_checkpoint(
        str(out),
        controller,
        include_fact_vectors=True,
        include_sentence_memory=True,
        sentence_compress="auto",
        fp16=True,
    )

    realms = getattr(controller.language, "realm_vectors", {}) or {}
    if realms and not args.no_realm_vectors:
        realm_names = sorted(realms.keys())
        print(
            f"  Realms:     {len(realm_names)} ({', '.join(realm_names[:12])}{'...' if len(realm_names) > 12 else ''})"
        )

        # Optional: build MoE router over realm indicators
        realm_fns = {r: (lambda x, _r=r: x) for r in realm_names}
        moe = ResonatorMoE.from_realm_vectors(
            dim=args.dim,
            realm_expert_fns=realm_fns,
            realm_vectors=None,  # hash-based routing stability
        )
        # Quick sanity routing
        ok = 0
        for r in realm_names:
            indicator = BinaryOps.hash_to_bipolar(r, args.dim)
            routed = moe.route(indicator, top_k=1)
            ok += int(routed[0] == r)
        print(f"  MoE route:  {ok}/{len(realm_names)} realm indicators matched")
    else:
        print("  Realms:     none")

    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
