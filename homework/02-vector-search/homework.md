## Homework: Vector Search (DONE)

In this homework, we put what we learned in Module 2 into practice.

We'll first turn text into vectors, then search by similarity.
We'll also learn something new and see how to combine vector search with keyword search. We'll skip the RAG part and focus solely on search.

Like in homework 1, our knowledge base is the course lessons themselves.
Each module has a `lessons/` folder of numbered markdown
pages, and we pull them from GitHub. We use the same commit, `8c1834d`,
so everyone works with the exact same 72 pages.

> It's possible your answers won't match exactly. If so, select the closest one.

## Setup

In this homework we won't use the same approach for embedding as in the
module. That is, we won't use the sentence-transformers library. Instead,
we'll use the lightweight embedding approach with the ONNX `Embedder`.

Both approaches produce identical vectors, but the ONNX runtime is far
lighter. It needs no PyTorch and no CUDA, which makes the install about
30x smaller and lets it run anywhere, including a basic Codespace. We
skimmed through it in the lesson and said we'd cover it in the homework -
so here we are.

We prepare the environment the same way as in the module's
[ONNX Runtime](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/lessons/09-onnx-embedder.md)
lesson.

Create a fresh project and install the dependencies:

```bash
mkdir llm-zoomcamp-hw2 && cd llm-zoomcamp-hw2
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch gitsource
uv add --dev huggingface-hub jupyter
```

We also need two helper scripts from the `embed/` directory of the course
repo:

- [`download.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/download.py)
(fetches an ONNX model from HuggingFace) and
- [`embedder.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/embedder.py) (the `Embedder` class with an `encode` interface)

Let's download them:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/02-vector-search/embed
wget $PREFIX/download.py
wget $PREFIX/embedder.py
```

By default `download.py` fetches `Xenova/all-MiniLM-L6-v2`, the ONNX
version of the `all-MiniLM-L6-v2` model from the lessons:

```bash
uv run python download.py
```

Now we're ready to do the homework.

## Q1. Embedding a query

Embed the following query:

> How does approximate nearest neighbor search work?

The embedder returns a vector of 384 numbers. What's the first value
(`v[0]`)?

* -0.31
* -0.02 <-- The Answer
* 0.12
* 0.44

### Q1 Answer
To answer Question 1, we need to set up the **ONNX Embedder** environment and run a specific script to extract the first value of the resulting vector. 

We can use the following Python code:

```python
import numpy as np
from embedder import Embedder

# Initialize the embedder
# Ensure the model has been downloaded to the default 'models' folder using download.py
embedder = Embedder(path="models/Xenova/all-MiniLM-L6-v2")

# Define the query from the homework
query = "How does approximate nearest neighbor search work?"

# Embed the query to get the vector
v = embedder.encode(query)

# Print the first value (v)
print(f"The first value (v) is: {v[0]:.2f}")
```

#### **Answer**
The first value of the vector (`v`) is approximately **-0.02**


## Loading the data

We pull the lesson pages from the course repository, the same way as in
homework 1. We pin to commit `8c1834d` so everyone works with the same
data.

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
```

Each document is a dictionary with a `filename` and `content`, and there
are 72 pages.

## Q2. Cosine similarity

The embedder returns normalized vectors, so the dot product between two
of them is their cosine similarity.

Take the page `02-vector-search/lessons/07-sqlitesearch-vector.md`, embed
its `content`, and compute the cosine similarity with the query vector
from Q1. What do you get?

* 0.07
* 0.37 <-- The Answer
* 0.68
* 0.92

### Q2 Answer
To answer Question 2, we will need to use the query vector we generated in Q1 and compare it against the content of the  page `02-vector-search/lessons/07-sqlitesearch-vector.md`. 

The **dot product** between them is mathematically equivalent to their **cosine similarity**.

We can use the following Python code, assuming we have already loaded our `documents` and initialized our `embedder`:

```python
# Initialize the reader to fetch markdown files from the /lessons/ folders
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]

target_file = '02-vector-search/lessons/07-sqlitesearch-vector.md'
target = [d for d in documents if d["filename"] == target_file][0]

page_vec = embedder.encode(target["content"])

# Cosine similarity
similarity = page_vec.dot(v)  # v is our Q1 query vector
print(f"The cosine similarity is: {similarity:.3f}")
```

#### **Answer**
The result is 0.36, this indicates a strong semantic similarity. The closest answer is **0.37**.

## Q3. Chunking and search by hand

A full page covers several topics, which waters down its embedding.

We chunk the pages the same way as in homework 1:

```python
from gitsource import chunk_documents
chunks = chunk_documents(documents, size=2000, step=1000)
```

We embed every chunk's `content` with `encode_batch`, stack the vectors
into a matrix `X`, and score the Q1 query against all chunks:

```python
scores = X.dot(v)
```

Which file does the highest-scoring chunk belong to (its `filename`)?

* `02-vector-search/lessons/03-embeddings-dataset.md`
* `02-vector-search/lessons/06-rag-vector.md`
* `02-vector-search/lessons/07-sqlitesearch-vector.md` <-- The Answer
* `02-vector-search/lessons/09-onnx-embedder.md`

### Q3 Answer
To answer Question 3, we need to process the course documents by breaking them into smaller segments (chunks), embedding those segments, and then identifying which segment is most semantically similar to our query from Q1.

We can use the following Python code:

```python
# Import library
from gitsource import chunk_documents

# Apply chunking with a 2000-character window and 1000-character step
# This will split the 72 pages into smaller pieces
chunks = chunk_documents(documents, size=2000, step=1000)

# Count the total number of resulting chunks
print(f"Number of chunks: {len(chunks)}")

# Embed every chunk and stack them into a matrix X
chunk_texts = [c["content"] for c in chunks]
X = embedder.encode_batch(chunk_texts)

# Compute similarity scores
scores = X.dot(v)

# Find the filename of the highest-scoring chunk
best_idx = scores.argmax()
best_chunk = chunks[best_idx]

print(f"Highest Score: {scores[best_idx]:.3f}")
print(f"Filename: {best_chunk['filename']}")
```

#### **Answer**
The highest-scoring chunk belongs to **`02-vector-search/lessons/07-sqlitesearch-vector.md`** with the score of **0.649**.

## Q4. Vector search with minsearch

We've done vector search by hand, which is good for learning, but it's not
what we do in practice. In practice we use libraries.

Let's use `VectorSearch` from minsearch and run a search for the following
query:

> What metric do we use to evaluate a search engine?

Which file is the `filename` of the first result?

* `02-vector-search/lessons/04-vector-search.md`
* `04-evaluation/lessons/05-search-metrics.md` <-- The Answer
* `04-evaluation/lessons/13-llm-as-judge.md`
* `05-monitoring/lessons/04-metrics.md`

### Q4 Answer
To answer Question 4, we will transition from manual vector search to using the **minsearch** library. 

We can use the following Python code, assumes we have already created the `chunks` and the matrix `X` (the embeddings for those chunks).

```python
from minsearch import VectorSearch

# Initialize VectorSearch
vector_index = VectorSearch(keyword_fields=["filename"])

# Index the chunks and their vectors
# X is the matrix of embeddings from Q3; chunks is the list of chunk dictionaries
vector_index.fit(X, chunks)

# Embed the new homework query
query2 = "What metric do we use to evaluate a search engine?"
v2 = embedder.encode(query2)

# Perform the vector search for top 5 results
results = vector_index.search(v2, num_results=5)
for i in results:
    print(i["filename"])

# Top result
print(f"\nThe first result for this query: {results[0]['filename']}")
```

#### **Answer**
The first result for this query is: **`04-evaluation/lessons/05-search-metrics.md`**.

## Q5. Text search vs vector search

Vector search matches by meaning, keyword search by exact words.

Let's compare them. Index
the same chunks with `Index` from minsearch. Use `content` as a
text field.

Run both searches for this query:

> How do I store vectors in PostgreSQL?

Take the top 5 results from each method. Which file shows up in the
vector results but not in the text results?

* `02-vector-search/lessons/01-intro.md`
* `02-vector-search/lessons/02-embeddings.md`
* `02-vector-search/lessons/08-pgvector.md` <-- The Answer
* `03-orchestration/lessons/05-rag.md`

### Q5 Answer
To answer Question 5, we need to compare the retrieval logic of a **Keyword (Text) Index** versus a **Vector Index** using the `minsearch` library.

We can use the following Python code:

```python
from minsearch import Index

# Setup the Text (Keyword) Index
# Using 'content' as the searchable text field
text_index = Index(
    text_fields=["content"], 
    keyword_fields=["filename"]
)
text_index.fit(chunks)

# Define the query and encode it for vector search
query3 = "How do I store vectors in PostgreSQL?"
v3 = embedder.encode(query3)

# Perform both searches
vector_results = vector_index.search(v3, num_results=5)
text_results = text_index.search(query3, num_results=5)

# Compare the resulting filenames
vector_filenames = {r["filename"] for r in vector_results}
text_filenames = {r["filename"] for r in text_results}

print("Vector Search Top 5:", vector_filenames)
print("Text Search Top 5:", text_filenames)

# Find the file in Vector results but not Text results
only_in_vector = vector_filenames - text_filenames
print(f"File in vector but not text: {only_in_vector}")
```

#### **Answer**
The file that shows up in the vector results but not in the text results is **`02-vector-search/lessons/08-pgvector.md`**.

## Q6. Hybrid search

Both vector and text search have their strengths and weaknesses. Vector
search matches by meaning, so it finds relevant pages even when they use
words different from the query. But it can miss exact terms like names,
codes, or rare keywords. Text search is the opposite: it nails exact words
but misses paraphrases and synonyms.

We don't have to pick one or the other - we can use both and merge their
results. This approach is called "hybrid search".

Each search produces its own ranked list, so we need a way to combine them
into one. In this homework we use Reciprocal Rank Fusion (RRF). It ignores
the raw scores from each method, which live on different scales and aren't
directly comparable. Instead, it looks only at the position of each
document in each list.

Every document scores by its position (`rank`, starting at 0) in each
list, and we sum the scores across lists with a constant `k = 60`:

```text
RRF(d) = sum over lists of  1 / (k + rank(d))
```

"Sum over lists" means we go through every ranked list and, for each list
where the document appears, add its `1 / (k + rank)` contribution. A
document found by both searches collects a score from each list, while one
found by only a single search collects just one.

The constant `k` controls how much the exact rank matters. A larger `k`
flattens the gap between positions, so the difference between rank 0 and
rank 5 counts for less. A smaller `k` does the opposite: it sharpens that
gap, so being at the top of a list matters much more.

The value 60 comes from the original RRF paper and is the usual default.
You rarely need to tune it. Lower it when only the top results matter.
Raise it to reward documents that appear across many lists, even when they
never quite reach the top.

A document that ranks well in both lists ends up higher than one that's
only strong in a single list.

```python
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]
```

Now run the query `"How do I give the model access to tools?"`
with vector and text search and fuse the results with `rrf`:

```python
results = rrf([vector_results, text_results])
```

Which file is ranked first after RRF?

* `01-agentic-rag/lessons/01-intro.md`
* `01-agentic-rag/lessons/13-function-calling.md` <-- The Answer
* `01-agentic-rag/lessons/14-agentic-loop.md`
* `01-agentic-rag/lessons/16-other-frameworks.md`

### Q6 Answer
To answer **Question 6** regarding **Hybrid Search**, we need to combine the strengths of both vector and keyword search using the **Reciprocal Rank Fusion (RRF)** algorithm.

We can use the following Python code:

```python
#  Define the RRF function provided in the homework
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

# Execute searches for the query
query4 = "How do I give the model access to tools?"
v4 = embedder.encode(query4)

# Perform both searches with Top 5 results
text_results_q6 = text_index.search(query4, num_results=5)
vector_results_q6 = vector_index.search(v4, num_results=5)

# Fuse results (text, vector)
hybrid_results = rrf([text_results_q6, vector_results_q6])

# Output the filename of the first result
print(f"\nTop Hybrid Result: {hybrid_results[0]['filename']}")
```

#### **Answer**

After running the hybrid search with the **all-MiniLM-L6-v2** model and the provided RRF implementation, the file ranked first is **`01-agentic-rag/lessons/13-function-calling.md`**.

## **Notebook**
For the complete work, you can review my notebook: [hw_02.ipynb](code/hw_02.ipynb) in the homework-02 [code folder](code).

**Thank you.**