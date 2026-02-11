"""
Example: Instant Language Learning

Demonstrates word encoding, sentence composition, parsing, and generation
without any training required.
"""

from grilly.experimental.language import (
    InstantLanguage,
    ResonatorParser,
    SentenceEncoder,
    SentenceGenerator,
    WordEncoder,
)

print("=" * 60)
print("Instant Language Learning Examples")
print("=" * 60)

dim = 2048

# Word Encoder
print("\n1. Word Encoding")
print("-" * 60)

word_encoder = WordEncoder(dim=dim)

cat_vec = word_encoder.encode_word("cat")
dog_vec = word_encoder.encode_word("dog")
print(f"Encoded 'cat': shape={cat_vec.shape}")
print(f"Encoded 'dog': shape={dog_vec.shape}")

similarity = word_encoder.similarity("cat", "dog")
print(f"Similarity between 'cat' and 'dog': {similarity:.4f}")

# Sentence Encoder
print("\n2. Sentence Encoding")
print("-" * 60)

sentence_encoder = SentenceEncoder(word_encoder)

words = ["the", "cat", "chased", "the", "mouse"]
sentence_vec = sentence_encoder.encode_sentence(words)
print(f"Sentence: {' '.join(words)}")
print(f"Encoded sentence: shape={sentence_vec.shape}")

role_vec = sentence_encoder.query_role(sentence_vec, "SUBJ")
print(f"Subject role query vector: {role_vec[:5]}")
role_words = word_encoder.find_closest(role_vec, top_k=3)
print(f"Closest words for SUBJ role: {role_words}")

# Sentence Generator
print("\n3. Sentence Generation")
print("-" * 60)

generator = SentenceGenerator(sentence_encoder)

template = generator.generate_from_roles({"SUBJ": "dog", "VERB": "barked", "OBJ": "loudly"})
print(f"Generated sentence: {' '.join(template)}")

relation_sentence = generator.generate_from_relation("cat", "chases", "mouse")
print(f"Relation sentence: {relation_sentence}")

# Resonator Parser
print("\n4. Sentence Parsing")
print("-" * 60)

parser = ResonatorParser(sentence_encoder, max_iterations=30)

parsed = parser.parse(sentence_vec)
print("Parsed sentence:")
for word, role, conf in parsed:
    print(f"  {word}: {role} (conf={conf:.2f})")

# Instant Language System
print("\n5. Instant Language System")
print("-" * 60)

lang = InstantLanguage(dim=dim)

lang.learn_sentence("the cat chased the mouse")
lang.learn_sentence("the dog barked loudly")
lang.learn_sentence("the bird flew high")

print("Learned sentences:")
for _, words in lang.sentence_memory:
    sentence = " ".join(words)
    print(f"  {sentence}")

# Query relations
result = lang.query_relation("cat", "chased")
print("\nQuery: What did the cat chase?")
print(f"Results: {result}")

# Parse sentence
parsed_sent = lang.parse_sentence("the dog barked loudly")
print(f"\nParsed: {parsed_sent}")

# Find similar sentences
similar = lang.find_similar_sentences("the cat ran fast", top_k=2)
print("\nSimilar sentences:")
for sent, sim in similar:
    print(f"  {sent}: {sim:.4f}")

# Analogy
print("\n6. Analogy")
print("-" * 60)

lang.learn_relation("king", "is_to", "queen")
lang.learn_relation("man", "is_to", "woman")

analogy = lang.analogy("king", "queen", "man")
print("Analogy: king:queen :: man:?")
print(f"Answer: {analogy}")
