#!/usr/bin/env python3
"""
SVC Ingestion Script - Load SVC JSONL data into the Grilly experimental pipeline.

Usage:
    python -m grilly.scripts.ingest_svc --file datasets/_data/instruct_svc_semantic.jsonl
    python -m grilly.scripts.ingest_svc --file datasets/_data/instruct_svc_semantic.jsonl --max 1000 --realms health science
    python -m grilly.scripts.ingest_svc --file datasets/_data/conversations_svc_semantic.jsonl --verbose

This script:
1. Loads SVC entries from a JSONL file
2. Ingests them into InstantLanguage (vocabulary, sentences, templates, realm vectors)
3. Ingests them into CognitiveController (world model facts, causal links)
4. Builds realm-routed ResonatorMoE expert vectors
5. Prints summary statistics
"""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SVC JSONL data into the Grilly experimental pipeline."
    )
    parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to the SVC JSONL file.",
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=None,
        help="Maximum number of entries to load.",
    )
    parser.add_argument(
        "--realms", "-r",
        nargs="*",
        default=None,
        help="Only load entries from these realms.",
    )
    parser.add_argument(
        "--min-complexity",
        type=float,
        default=None,
        help="Minimum complexity threshold.",
    )
    parser.add_argument(
        "--max-complexity",
        type=float,
        default=None,
        help="Maximum complexity threshold.",
    )
    parser.add_argument(
        "--sources", "-s",
        nargs="*",
        default=None,
        help="Only load entries from these sources (instruct, conversation).",
    )
    parser.add_argument(
        "--dim", "-d",
        type=int,
        default=2048,
        help="VSA vector dimension (default: 2048).",
    )
    parser.add_argument(
        "--no-templates",
        action="store_true",
        help="Skip template learning.",
    )
    parser.add_argument(
        "--no-realm-vectors",
        action="store_true",
        help="Skip building realm vectors.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress.",
    )
    args = parser.parse_args()

    # Imports here so --help is fast
    from grilly.experimental.language.svc_loader import load_svc_batch, SVCIngestionEngine
    from grilly.experimental.language.system import InstantLanguage
    from grilly.experimental.cognitive.controller import CognitiveController
    from grilly.experimental.moe.routing import ResonatorMoE

    print("=" * 60)
    print("Grilly SVC Ingestion Pipeline")
    print("=" * 60)

    # ---- Step 0: Create engine (auto-detect GPU) ----
    engine = SVCIngestionEngine(dim=args.dim)
    print(f"\nEngine: {engine.status()}")

    # ---- Step 1: Load data ----
    print(f"\n[1/4] Loading SVC data from: {args.file}")
    t0 = time.time()

    try:
        batch = load_svc_batch(
            path=args.file,
            max_entries=args.max,
            realms=args.realms,
            min_complexity=args.min_complexity,
            max_complexity=args.max_complexity,
            sources=args.sources,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    load_time = time.time() - t0
    print(f"  Loaded {batch.total_loaded} entries in {load_time:.2f}s")
    if args.verbose:
        print(batch.summary())

    if not batch.entries:
        print("No entries to ingest. Exiting.")
        sys.exit(0)

    # ---- Step 2: Language ingestion (GPU-accelerated) ----
    print(f"\n[2/4] Ingesting into InstantLanguage (dim={args.dim})...")
    t1 = time.time()

    lang = InstantLanguage(dim=args.dim)
    lang_result = lang.ingest_svc(
        batch.entries,
        learn_templates=not args.no_templates,
        build_realm_vectors=not args.no_realm_vectors,
        verbose=args.verbose,
        engine=engine,
    )

    lang_time = time.time() - t1
    print(f"  {lang_result.sentences_learned} sentences, "
          f"{lang_result.words_encoded} new words in {lang_time:.2f}s")
    print(f"  {lang_result.templates_learned} templates learned")
    print(f"  {len(lang_result.realm_vectors)} realm vectors built")
    print(f"  Backend: {lang_result.backend}")

    # ---- Step 3: Cognitive ingestion (GPU-accelerated) ----
    print(f"\n[3/4] Ingesting into CognitiveController...")
    t2 = time.time()

    controller = CognitiveController(dim=args.dim)
    cog_result = controller.ingest_svc(
        batch.entries,
        learn_templates=not args.no_templates,
        build_realm_vectors=not args.no_realm_vectors,
        verbose=args.verbose,
        engine=engine,
    )

    cog_time = time.time() - t2
    print(f"  {len(controller.world.facts)} world facts added in {cog_time:.2f}s")
    print(f"  {len(controller.world.expectations)} causal links")

    # ---- Step 4: Realm MoE ----
    print(f"\n[4/4] Building realm-routed MoE...")
    realms = lang_result.realm_vectors
    if realms:
        realm_fns = {r: (lambda x, _r=r: x) for r in realms}
        moe = ResonatorMoE.from_realm_vectors(
            dim=args.dim,
            realm_expert_fns=realm_fns,
            realm_vectors=None,  # Use hash-based for routing stability
        )
        print(f"  ResonatorMoE with {len(realms)} realm experts: {sorted(realms.keys())}")

        # Quick routing test
        from grilly.experimental.vsa.ops import BinaryOps
        for realm in sorted(realms.keys()):
            indicator = BinaryOps.hash_to_bipolar(realm, args.dim)
            routed = moe.route(indicator, top_k=1)
            status = "OK" if routed[0] == realm else f"MISMATCH ({routed[0]})"
            print(f"    {realm} -> {status}")
    else:
        print("  No realm vectors built (skipped or no data).")

    # ---- Summary ----
    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {total_time:.2f}s total")
    print(f"  Entries: {batch.total_loaded}")
    print(f"  Words:   {lang_result.words_encoded}")
    print(f"  Facts:   {len(controller.world.facts)}")
    print(f"  Realms:  {sorted(realms.keys()) if realms else 'none'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
