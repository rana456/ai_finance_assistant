# Beyond Vector Search: Building Hybrid RAG for a Financial Q&A Assistant

### How combining FAISS embeddings, BM25, and Reciprocal Rank Fusion — plus one grounding trick — made our retrieval actually trustworthy

---

When I set out to build the knowledge engine for an AI finance assistant, the first instinct was the obvious one: chunk the documents, embed them, drop them in a vector database, and let cosine similarity do the work. That's the textbook RAG pipeline, and it works — until it doesn't.

The moment it stopped working was a question like this:

> *"What's an expense ratio?"*

Pure vector search kept surfacing chunks about "annual fees" and "fund costs" — semantically related, sure, but it repeatedly ranked a vague paragraph *above* the article literally titled **Expense Ratios**. For a financial education tool, that's not a rounding error. Beginners ask about precise terms — *401(k)*, *P/E ratio*, *Roth IRA* — and getting the exact right source matters.

The fix was **hybrid retrieval**: combining semantic (dense) search with keyword (sparse) search, fusing the two rankings, and adding a grounding signal so the system knows when to say *"I don't have that."* This article walks through the whole thing, with the real code.

---

## First, a quick refresher: what RAG actually solves

A large language model only knows what was in its training data, and it will confidently invent an answer when asked something it doesn't know. For general chit-chat that's tolerable. For *"explain how a Roth IRA is taxed,"* a hallucinated answer is actively harmful.

Retrieval-Augmented Generation (RAG) fixes this by giving the model the relevant source material **at question time**, so it answers from documents you control instead of from memory. Two phases:

**Indexing (offline):**
```
articles → chunk into passages → embed each chunk → store in an index
```

**Retrieval + generation (per question):**
```
question → find the most relevant chunks → hand them to the LLM
         → "answer using ONLY these sources, and cite them"
```

Everything interesting happens in that middle step — *finding the most relevant chunks*. That's where dense and sparse retrieval diverge.

---

## Two ways to match a question to a chunk

There are two fundamentally different ways to decide whether a passage answers a query.

### Dense retrieval (semantic / vector search)

You convert both the query and each chunk into a high-dimensional vector using an embedding model, then rank chunks by cosine similarity. Because embeddings capture **meaning**, this handles paraphrase beautifully:

> *"How do I spread my stock purchases out over time?"* → matches a chunk about **dollar-cost averaging**, even though they share no words.

Its weakness is the flip side of its strength: it can *blur* exact terms. Precise strings like `401(k)`, `expense ratio`, or a specific ticker get smeared into their semantic neighborhood, and a vaguely-related chunk can outrank the exact one.

### Sparse retrieval (lexical / BM25)

This is the classic search-engine approach. **BM25** scores a chunk by how many query terms it contains, weighted by how rare those terms are and normalized for document length. It nails exact terminology and acronyms — but it's literal. It won't connect a paraphrase to the right concept, and it's blind to synonyms.

### The finance-domain case for using both

Here's the table that convinced me. Financial content is full of **precise jargon that also has plain-language paraphrases** — exactly the case where either method alone leaves gold on the table:

| User asks | Dense catches | Sparse (BM25) catches |
|---|---|---|
| "what's an expense ratio" | — | **exact term** ✓ |
| "the yearly fee funds charge" | **concept** ✓ | — |
| "difference between a 401k and IRA" | concept ✓ | **both acronyms** ✓ |

Hybrid retrieval gets all three rows right. That's the entire argument.

---

## Combining them with Reciprocal Rank Fusion

The obvious question: if I run both retrievers, how do I merge their results? Cosine similarities live on a different scale than BM25 scores — you can't just add them. You *could* normalize both to [0, 1] and take a weighted sum, but then you're hand-tuning weights and normalization schemes forever.

**Reciprocal Rank Fusion (RRF)** sidesteps all of that. Instead of combining *scores*, it combines *ranks*:

```
score(d) = Σ  1 / (k + rank_r(d))
          r
```

For each document `d`, you sum a contribution from every retriever `r` it appears in, where `rank_r(d)` is its position in that retriever's list (0-indexed). The constant `k` (conventionally **60**) dampens the influence of exact rank, so *"in the top handful of either list"* matters more than *"#1 vs #2."*

The elegance: RRF needs **no score normalization and no tuning**. A chunk that both methods rank highly floats to the top; a chunk only one method likes still gets a fair shot. Here's the actual fusion loop from our retriever:

```python
RRF_K = 60  # standard default; dampens the influence of exact rank

# dense_rank / sparse_rank: {chunk_index: rank} from each retriever
candidates = set(dense_rank) | set(sparse_rank)

fused = []
for idx in candidates:
    rrf = 0.0
    if idx in dense_rank:
        rrf += 1.0 / (RRF_K + dense_rank[idx])
    if idx in sparse_rank:
        rrf += 1.0 / (RRF_K + sparse_rank[idx])
    fused.append(ScoredChunk(chunk=chunks[idx], rrf=rrf, ...))

fused.sort(key=lambda sc: sc.rrf, reverse=True)
top = fused[:top_k]
```

That's it. No weights to tune, no scales to reconcile.

---

## The "I don't know" problem — and why RRF can't solve it

Here's a subtlety that trips people up. A financial assistant **must** be able to say *"I don't have information on that"* rather than confidently answering from loosely-related chunks. Ask it *"what's the best recipe for lasagna?"* and it should decline, not improvise from a diversification article.

You might think the fusion score handles this — surely an irrelevant query produces low scores? **It doesn't.** RRF is a measure of *agreement between retrievers*, not of *relevance to the query*. Both BM25 and the vector search will always return *something* as their top result, so the top RRF score is always non-trivial, even for lasagna.

So RRF is the wrong signal for grounding. What we need is a **semantic floor**: is the single best chunk actually similar enough to the query to be worth answering from? Cosine similarity is exactly that signal — it's absolute (0 to 1), not relative to other chunks. So we use **RRF for ordering** and **cosine for the grounding decision**:

```python
DEFAULT_GROUNDING_THRESHOLD = 0.28  # tuned for text-embedding-3-small

# ... after fusion, `top` is the RRF-ranked shortlist ...
is_grounded = bool(top) and max(sc.cosine for sc in top) >= self.grounding_threshold
return RetrievalResult(results=top, is_grounded=is_grounded)
```

If the best chunk's cosine similarity clears the threshold, we answer and cite. If not, the agent responds with a graceful *"that's not in my knowledge base — here's what I can help with."* This one line is the difference between a system that hallucinates and one that knows its limits.

> **Note on the two signals:** every candidate chunk carries *both* a `cosine` (semantic relevance, our grounding floor) and an `rrf` (fusion rank, our ordering). Keeping them separate is what lets RRF rank while cosine gates.

---

## The full retriever

Putting it together — dense search, sparse search, RRF fusion, an optional category filter, and the grounding gate:

```python
from rank_bm25 import BM25Okapi

RRF_K = 60
DEFAULT_GROUNDING_THRESHOLD = 0.28

class HybridRetriever:
    def __init__(self, vector_store, embedder, grounding_threshold=DEFAULT_GROUNDING_THRESHOLD):
        self.store = vector_store
        self.embedder = embedder
        self.grounding_threshold = grounding_threshold
        # Build the BM25 index once, over the same chunks as the vector store.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in vector_store.chunks])

    def retrieve(self, query, top_k=4, category=None):
        # Pull a wider candidate pool than top_k so fusion has room to work.
        pool = max(top_k * 3, 10)

        # --- Dense: FAISS cosine similarity ---
        query_embedding = self.embedder.embed_query(query)
        dense = self.store.search(query_embedding, pool)          # [(idx, cosine)]
        dense_rank = {idx: rank for rank, (idx, _) in enumerate(dense)}

        # --- Sparse: BM25 ---
        scores = self._bm25.get_scores(_tokenize(query))
        sparse_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sparse_idxs = [i for i in sparse_idxs if scores[i] > 0][:pool]
        sparse_rank = {idx: rank for rank, idx in enumerate(sparse_idxs)}

        # --- Reciprocal Rank Fusion ---
        fused = []
        for idx in set(dense_rank) | set(sparse_rank):
            rrf = 0.0
            if idx in dense_rank:  rrf += 1.0 / (RRF_K + dense_rank[idx])
            if idx in sparse_rank: rrf += 1.0 / (RRF_K + sparse_rank[idx])
            chunk = self.store.chunks[idx]
            if category is not None and chunk.category != category:
                continue
            cosine = self.store.cosine_for(idx, query_embedding)
            fused.append(ScoredChunk(chunk=chunk, cosine=max(0.0, cosine), rrf=rrf))

        fused.sort(key=lambda sc: sc.rrf, reverse=True)
        top = fused[:top_k]

        is_grounded = bool(top) and max(sc.cosine for sc in top) >= self.grounding_threshold
        return RetrievalResult(results=top, is_grounded=is_grounded)
```

A few implementation notes worth calling out.

### FAISS with a flat index = exact cosine

We store L2-normalized embeddings in a FAISS `IndexFlatIP` (inner-product) index. Because the vectors are normalized, **inner product equals cosine similarity**. And because the index is *flat* (exhaustive), search is **exact** — no approximate-nearest-neighbor recall loss.

For a large corpus you'd reach for an approximate index (IVF, HNSW). But our knowledge base is a few hundred curated chunks, and at that scale flat search is instant and lossless. Reaching for ANN here would add tuning knobs and recall risk to solve a problem we don't have. **Match the index to the corpus size.**

```python
import faiss, numpy as np

def build(chunks, embeddings):
    mat = normalize(np.asarray(embeddings, dtype="float32"))  # L2-normalize rows
    index = faiss.IndexFlatIP(mat.shape[1])                   # inner product = cosine
    index.add(mat)
    return VectorStore(index, chunks, mat)
```

### Chunking on headings

Our source articles are short and section-structured (markdown with `##` headings), so we chunk on headings and prefix each chunk with its article title and section. That keeps every chunk **self-describing** — a retrieved passage carries enough context to stand on its own, out of its document.

### Embeddings

We use OpenAI's `text-embedding-3-small` (1536 dimensions, cheap). The grounding threshold of `0.28` is calibrated to *that* model's cosine distribution — if you swap embedders, recalibrate it.

---

## What it looks like in practice

Three queries, three different behaviors, all correct:

- **"How does compound interest work?"** → dense and sparse agree, RRF ranks the *Compound Interest* article first, cosine well above threshold → grounded, cited answer.
- **"What's an expense ratio?"** → BM25 catches the exact term that dense search was blurring, RRF pulls the exact article up → grounded.
- **"What's the best recipe for lasagna?"** → both retrievers return their top guesses, but the best cosine is far below `0.28` → **not grounded**, the assistant declines instead of hallucinating.

That last one is the payoff. The lasagna query *still produces a non-trivial RRF score* — which is exactly why we don't ground on RRF.

---

## Takeaways

If you're building RAG over a domain with precise terminology — finance, law, medicine, engineering — here's what I'd distill:

1. **Pure vector search under-serves exact terms.** If your users type jargon, acronyms, or codes, add sparse retrieval. The two are complementary, not redundant.
2. **RRF is the cheapest good fusion method.** Rank-based, no score normalization, no weights to tune. Start here before anything fancier.
3. **Fusion score ≠ relevance.** RRF measures agreement, not relevance, so it can't tell you when to abstain. Use an absolute signal — cosine similarity — as a separate grounding floor.
4. **Right-size the index.** Flat/exact search is a feature, not a limitation, for small curated corpora. Save the ANN complexity for when you actually have the scale.
5. **Grounding is a product decision, not just a technical one.** The `is_grounded` flag is what lets the system say *"I don't know"* — and in high-stakes domains, that honesty is the whole point.

Hybrid retrieval isn't exotic, and none of these pieces are hard on their own. But wiring them together — dense for meaning, sparse for precision, RRF to merge, cosine to abstain — turned a retriever that confidently returned *almost-right* answers into one I'd actually trust to teach someone about their money.

---

*The full implementation lives in the [AI Finance Assistant](#) project — a multi-agent financial education system with five specialized agents, LangGraph orchestration, and this hybrid RAG at its core. All code is open source.*
