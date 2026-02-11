def test_checkpoint_sentence_token_compression(tmp_path):
    try:
        from grilly.experimental.cognitive.controller import CognitiveController
        from grilly.experimental.language.svc_loader import SVCEntry
    except ModuleNotFoundError:
        from experimental.cognitive.controller import CognitiveController
        from experimental.language.svc_loader import SVCEntry

    try:
        from grilly.utils.ingest_checkpoint import (
            CheckpointView,
            load_ingest_checkpoint,
            save_ingest_checkpoint,
        )
    except ModuleNotFoundError:
        from utils.ingest_checkpoint import (
            CheckpointView,
            load_ingest_checkpoint,
            save_ingest_checkpoint,
        )

    c = CognitiveController(dim=128, word_use_ngrams=False)

    entries = [
        SVCEntry(
            id="t1",
            text="We modified your code.",
            svc_s="we",
            svc_v="modify",
            svc_c="your code",
            pos=["PRON", "VERB", "PRON", "NOUN"],
            lemmas=["we", "modify", "your", "code"],
            deps=["nsubj", "ROOT", "poss", "dobj"],
            root_verb="modify",
            realm="technology",
            source="conversation",
            complexity=0.2,
        ),
        SVCEntry(
            id="t2",
            text="We modified the pipeline.",
            svc_s="we",
            svc_v="modify",
            svc_c="the pipeline",
            pos=["PRON", "VERB", "DET", "NOUN"],
            lemmas=["we", "modify", "the", "pipeline"],
            deps=["nsubj", "ROOT", "det", "dobj"],
            root_verb="modify",
            realm="technology",
            source="conversation",
            complexity=0.2,
        ),
    ]

    c.ingest_svc(
        entries, learn_templates=False, build_realm_vectors=False, verbose=False, engine=None
    )

    out = tmp_path / "ckpt.npz"
    save_ingest_checkpoint(
        str(out), c, include_sentence_memory=True, sentence_compress="auto", fp16=True
    )

    view = CheckpointView(str(out))
    assert view.sentence_count() >= 0
    toks0 = view.get_sentence_tokens(0)
    assert isinstance(toks0, list)
    assert len(toks0) > 0

    c2 = CognitiveController(dim=128, word_use_ngrams=False)
    man = load_ingest_checkpoint(str(out), c2)
    assert man["format"].endswith("v2")
    assert len(c2.world.facts) == len(c.world.facts)
