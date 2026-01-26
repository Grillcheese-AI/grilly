"""
Quick test script to verify dataset loading works correctly.
"""
import json
from grilly.examples.train_capsule_from_dataset import load_conversation_dataset, extract_text_pairs

# Test loading
print("Testing dataset loading...")
entries = load_conversation_dataset('grilly/examples/conversations_dataset_anonymized.jsonl', max_samples=10)
print(f"Loaded {len(entries)} entries")

if entries:
    print(f"\nFirst entry keys: {list(entries[0].keys())}")
    if 'messages' in entries[0]:
        print(f"Messages type: {type(entries[0]['messages'])}")
        if isinstance(entries[0]['messages'], list) and len(entries[0]['messages']) > 0:
            print(f"First message type: {type(entries[0]['messages'][0])}")
            if isinstance(entries[0]['messages'][0], dict):
                print(f"First message keys: {list(entries[0]['messages'][0].keys())}")

# Test extraction
print("\nTesting text pair extraction...")
anchors, positives, negatives = extract_text_pairs(entries)
print(f"Extracted {len(anchors)} anchor-positive-negative triplets")

if anchors:
    print(f"\nSample anchor (first 100 chars): {anchors[0][:100]}...")
    print(f"Sample positive (first 100 chars): {positives[0][:100]}...")
    print(f"Sample negative (first 100 chars): {negatives[0][:100]}...")

print("\nDataset loading test complete!")
