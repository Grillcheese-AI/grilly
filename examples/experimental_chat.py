"""
Interactive chat interface for experimental modules.

Default mode uses CognitiveController to generate responses.
"""

import argparse

from grilly.experimental.cognitive import CognitiveController


def parse_fact(text: str) -> tuple[str, str, str] | None:
    """
    Parse "/fact subject|relation|object".
    Returns tuple or None if invalid.
    """
    parts = text.split(" ", 1)
    if len(parts) != 2:
        return None
    payload = parts[1]
    fields = [p.strip() for p in payload.split("|")]
    if len(fields) != 3 or not all(fields):
        return None
    return fields[0], fields[1], fields[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental chat interface")
    parser.add_argument("--dim", type=int, default=2048, help="Vector dimension")
    parser.add_argument("--confidence", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--verbose", action="store_true", help="Show thinking trace")
    args = parser.parse_args()

    controller = CognitiveController(dim=args.dim, confidence_threshold=args.confidence)

    print("Interactive chat. Type /exit to quit.")
    print("Commands:")
    print("  /fact subject|relation|object")
    print("  /exit")

    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text.startswith("/fact"):
            fact = parse_fact(text)
            if fact is None:
                print("Invalid fact format. Use: /fact subject|relation|object")
                continue
            subj, rel, obj = fact
            controller.world.add_fact(subj, rel, obj)
            print("Fact added.")
            continue

        response = controller.process(text, verbose=args.verbose)
        if response is None:
            print("Assistant: (no response)")
        else:
            print(f"Assistant: {response}")

        if args.verbose:
            trace = controller.thinking_trace
            if trace:
                print("Trace:")
                for step in trace:
                    print(f"  {step}")


if __name__ == "__main__":
    main()
