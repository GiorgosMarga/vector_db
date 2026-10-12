import gensim.downloader as api

MODEL = "glove-wiki-gigaword-50"
OUTPUT = "word_vectors.txt"
LIMIT = 10_000

print(f"Loading {MODEL}...")
vectors = api.load(MODEL)

print(f"Writing first {LIMIT} words to {OUTPUT}...")

with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, word in enumerate(vectors.index_to_key[:LIMIT]):
        vec = vectors[word]

        values = " ".join(str(float(x)) for x in vec)

        f.write(f"{word} {values}\n")

print("Done.")
