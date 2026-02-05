"""
Example: Instant Language Learning

Demonstrates word encoding, sentence composition, parsing, and generation
without any training required.
"""

from grilly.experimental.language import (
    WordEncoder, SentenceEncoder, SentenceGenerator,
    ResonatorParser, InstantLanguage
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

roles = sentence_encoder.query_role(sentence_vec, "SUBJECT")
print(f"Subject role query: {roles[:5]}")

# Sentence Generator
print("\n3. Sentence Generation")
print("-" * 60)

generator = SentenceGenerator(sentence_encoder)

template = generator.generate_from_roles({
    "SUBJECT": "dog",
    "VERB": "barked",
    "OBJECT": "loudly"
})
print(f"Generated sentence: {template}")

relation_sentence = generator.generate_from_relation("cat", "chases", "mouse")
print(f"Relation sentence: {relation_sentence}")

# Resonator Parser
print("\n4. Sentence Parsing")
print("-" * 60)

parser = ResonatorParser(sentence_encoder, max_iterations=30)

parsed = parser.parse(sentence_vec)
print(f"Parsed sentence:")
for word, role in parsed:
    print(f"  {word}: {role}")

# Instant Language System
print("\n5. Instant Language System")
print("-" * 60)

lang = InstantLanguage(dim=dim)

lang.learn_sentence("the cat chased the mouse")
lang.learn_sentence("the dog barked loudly")
lang.learn_sentence("the bird flew high")

print("Learned sentences:")
for sentence in lang.language.sentence_encoder.word_encoder.vocabulary.keys():
    if len(sentence.split()) > 1:
        print(f"  {sentence}")

# Query relations
result = lang.query_relation("cat", "chased")
print(f"\nQuery: What did the cat chase?")
print(f"Results: {result}")

# Parse sentence
parsed_sent = lang.parse_sentence("the dog barked loudly")
print(f"\nParsed: {parsed_sent}")

# Find similar sentences
similar = lang.find_similar_sentences("the cat ran fast", top_k=2)
print(f"\nSimilar sentences:")
for sent, sim in similar:
    print(f"  {sent}: {sim:.4f}")

# Analogy
print("\n6. Analogy")
print("-" * 60)

lang.learn_relation("king", "is_to", "queen")
lang.learn_relation("man", "is_to", "woman")

analogy = lang.analogy("king", "queen", "man")
print(f"Analogy: king:queen :: man:?")
print(f"Answer: {analogy}")
