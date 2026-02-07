import io
from contextlib import redirect_stdout


def _two_entries():
    try:
        from grilly.experimental.language.svc_loader import SVCEntry
    except ModuleNotFoundError:  # compatibility for minimal zips
        from experimental.language.svc_loader import SVCEntry

    e1 = SVCEntry(
        id="t1",
        text="Below, I've modified your code.",
        svc_s="I",
        svc_v="modify",
        svc_c="your code",
        pos=["ADV", "PRON", "AUX", "VERB", "PRON", "NOUN"],
        lemmas=["below", "i", "have", "modify", "your", "code"],
        deps=["advmod", "nsubj", "aux", "ROOT", "poss", "dobj"],
        root_verb="modify",
        realm="technology",
        source="conversation",
        complexity=0.48,
    )
    e2 = SVCEntry(
        id="t2",
        text="The model predicts future events.",
        svc_s="model",
        svc_v="predict",
        svc_c="future events",
        pos=["DET", "NOUN", "VERB", "ADJ", "NOUN"],
        lemmas=["the", "model", "predict", "future", "event"],
        deps=["det", "nsubj", "ROOT", "amod", "dobj"],
        root_verb="predict",
        realm="science",
        source="instruct",
        complexity=0.40,
    )
    return [e1, e2]


def test_instantlanguage_ingest_no_spam_output():
    """ingest_svc must not print one line per entry when verbose=False."""
    try:
        from grilly.experimental.language.system import InstantLanguage
    except ModuleNotFoundError:
        from experimental.language.system import InstantLanguage

    lang = InstantLanguage(dim=512)
    entries = _two_entries()

    buf = io.StringIO()
    with redirect_stdout(buf):
        lang.ingest_svc(entries, learn_templates=False, build_realm_vectors=False, verbose=False)

    out = buf.getvalue()
    assert "Ingested sentence" not in out
    # A couple of header lines are fine, but avoid per-entry spam.
    assert out.count("\n") < 20
