## Homework: Agentic RAG (DONE)

In this homework, we build a RAG system from scratch and then make it
agentic - the same path as the module.

Instead of the course FAQ, our knowledge base is the course lessons
themselves.

The course repository is organized by module. Each module is a top-level
folder with a `lessons/` subfolder of numbered markdown pages:

```
01-agentic-rag/
└── lessons/
    ├── 01-intro.md
    ├── 02-environment.md
    ├── ...
    └── 16-other-frameworks.md
```

There are seven modules:

- `01-agentic-rag`
- `02-vector-search`
- `03-orchestration`
- `04-evaluation`
- `05-monitoring`
- `06-best-practices`
- `07-project-example`

Each lesson page is a single markdown file. These pages are exactly what you
read as you go through the course.

We'll fetch this data from GitHub and use it as the knowledge base for our
RAG system.

> It's possible your answers won't match exactly. If so, select the closest one.

## Setup

Prepare your environment the same way as in the module's
[Environment](../../modul/01-agentic-rag/lessons/02-environment.md) lesson.

This homework needs one extra library: `gitsource`, which downloads files
from a GitHub repository.

Install it:

```bash
uv add gitsource
```

For the LLM, we recommend OpenAI with `gpt-5.4-mini`, but you can use any model
and provider you like - just adapt the client and the usage fields accordingly.

## Preparation

First, we will pull the lesson pages straight from the course repository. 
We will use the commit `8c1834d` to make sure everyone works with the exact same data.

We will use `gitsource` for that:

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
```

`GithubRepositoryDataReader` downloads the entire repository and goes over all the files in it. Because we specify `allowed_extensions={"md"}`, it only checks  the markdown files.

We also pass a `filename_filter` so we don't grab every markdown file in the
repo, like the top-level README. The lesson pages all live under a module's `lessons/` folder, so
filtering on `/lessons/` keeps just those.


Each file has a `parse()` method that returns a dictionary with its
`filename` and `content`:

```python
documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)
```

## Q1. How many lesson pages

How many lesson pages are in the dataset?

* 24
* 72 <-- The Answer
* 240
* 720

### Q1 Answer

The course content is organized into seven modules, each containing a `lessons/` subfolder with numbered markdown files. To answer the first question, we need to count the number of markdown files that serve as lesson pages within the course repository.

We can use the following code to fetch the dataset and count the number of lesson pages:

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo="DataTalksClub/llm-zoomcamp",
    branch="main",
    commit="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda f: "/lessons/" in f
)

# Load the documents
files = reader.read()
documents = [f.parse() for f in files]

# Count the number of lesson pages (documents)
print(f"Number of lesson pages: {len(documents)}")
```

**Answer**: The correct choice is **72**

## Q2. Indexing and searching

Index the documents with minsearch - make `content` a text field and
`filename` a keyword field. Then search with this query:

> How does the agentic loop keep calling the model until it stops?

What's the `filename` of the first result?

* `01-agentic-rag/lessons/03-rag.md`
* `01-agentic-rag/lessons/14-agentic-loop.md` <-- The Answer
* `04-evaluation/lessons/13-llm-as-judge.md`
* `06-best-practices/lessons/02-hybrid-search.md`

### Q2 Answer
To answer this question, we need to index the lesson documents into the `minsearch` engine using the specified schema and then perform a search with the provided query.

We can use the following code to perform the indexing and search:

```python
import minsearch

# Index the documents with the specified schema
index = minsearch.Index(
    text_fields=['content'],
    keyword_fields=['filename']
)
index.fit(documents)

# Perform the search
query = "How does the agentic loop keep calling the model until it stops?"
search_results = index.search(query, num_results=1)

# 4. Extract the filename of the first result
print(f"First result filename: {search_results['filename']}")
```

#### **Answer**
The correct filename is: **01-agentic-rag/lessons/14-agentic-loop.md**

## Q3. RAG

Now we will build a RAG assistant on top of this data. Let's use the rag helper 
script we prepared during the lessons:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-agentic-rag/code/rag_helper.py
```

`RAGBase` was written for the FAQ schema (`section`/`question`/`answer`),
while our documents have `filename` and `content`.

Two solutions are possible:

- Implement the RAG flow yourself
- Take `RAGBase` and change the parts related to the FAQ schema - `search` (to use our index) and `build_context`

Build a RAG over the index from Q2 and answer the query:

> How does the agentic loop keep calling the model until it stops?

Use gpt-5.4-mini. How many input (prompt) tokens did we send to the model for
this request?

* 700
* 7000 <-- The Answer
* 70000
* 700000

We count input tokens instead of price because the cost depends on the model
and provider you use, but the size of the prompt we send is the same for
everyone.

Most LLM APIs report token usage on the response object (e.g.
`response.usage.input_tokens` / `prompt_tokens`). We'll read the input tokens
from there.

You will need to modify the code for the rag helper to expose the usage.

In the RAG Helper class, `llm` returns only the text. Modify it to return the whole response, and change `rag` to return both the answer and usage (as a tuple or create a small dataclass for that).

### Q3 Answer
To answer the third question, we need to modify the `RAGBase` class to handle the lesson dataset schema (`filename` and `content`) and expose the token usage from the LLM response.

We can implement the required changes by subclassing `RAGBase` or modifying the script directly as shown below:

```python
# Update the original `RAGBase` class 
from rag_helper import RAGBase
import dataclasses

@dataclasses.dataclass
class RAGResult:
    answer: str
    usage: object

class LessonRAG(RAGBase):

    def search(self, query, num_results=5):
        # no boost_dict/filter_dict here — your index has no
        # 'question'/'section'/'course' fields to boost or filter on
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"FILE: {doc['filename']}")
            lines.append(doc['content'])
            lines.append('')
        return '\n'.join(lines).strip()

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response  # full response object now, not just .output_text

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return RAGResult(answer=response.output_text, usage=response.usage)

# Run the query and print the result
rag = LessonRAG(index=index, llm_client=openai_client)
result = rag.rag("How does the agentic loop keep calling the model until it stops?")
print(result.usage)
```

#### **Answer**
The actual run resulting in input_tokens=7126. So the nearest answer is **7000**

## Q4. Chunking

The lesson pages are long - some are thousands of characters. Long documents
make retrieval less precise: a match deep inside a page still pulls in the
whole page. A common fix is chunking: split each page into smaller,
overlapping pieces and index those instead.

gitsource has a helper for this: `chunk_documents`. It uses a sliding
window - a window of `size` characters slides across the text in steps of
`step` characters, and each window position becomes one chunk:

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
```

With `size=2000` and `step=1000` (you can see the implementation
[here](https://github.com/alexeygrigorev/gitsource/blob/master/gitsource/chunking.py)):

- Each chunk is a window of `size` characters of the page.
- The window moves forward by `step` characters between chunks. Since `step`
  is smaller than `size`, consecutive chunks overlap by `size - step` (1000)
  characters, so a passage split across a boundary still appears whole in one
  of the chunks.
- Every chunk keeps the original fields (`filename`) and adds `start` (the
  offset in the page) and `content` (the chunk text).

How many chunks do you get?

* 70
* 295 <-- The Answer
* 1100
* 4500

### Q4 Answer
To answer the fourth question, we need to apply a **sliding window chunking strategy** to the lesson pages retrieved in Q1.

We can use the following code to chunk the 72 lesson pages fetched earlier:

```python
from gitsource import chunk_documents

# Apply chunking with a 2000-character window and 1000-character step
# This will split the 72 pages into smaller pieces
chunks = chunk_documents(documents, size=2000, step=1000)

# 3. Count the total number of resulting chunks
print(f"Number of chunks: {len(chunks)}")
```

#### **Answer**
The correct answer is **295**

## Q5. RAG with chunking

Chunking makes each request smaller, because we send a smaller context to the
LLM. Let's measure that.

Index the chunks from Q4 (same as before: `content` as a text field,
`filename` as a keyword field), point your RAG at the chunk index, and
answer the same query again - reading the input tokens the same way as in Q3.

Compare the input tokens with Q3. How many fewer input tokens does the chunked
version send?

* about the same
* 3× fewer <-- The Answer
* 10× fewer
* 30× fewer

### Q5 Answer
To answer Question 5, we need to transition our RAG assistant from searching full lesson pages to searching the overlapping chunks we created in Question 4.

We can use the following code:
```python
# Index the chunks in minsearch
# We treat the chunk objects as dictionaries for the index
chunk_index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)
chunk_index.fit(chunks)

# Initialize the RAG assistant with the chunk index
# Using the modified LessonRAG class from Q3
chunk_rag = LessonRAG(index=chunk_index, llm_client=openai_client)
result_q5 = chunk_rag.rag("How does the agentic loop keep calling the model until it stops?")
print(result_q5.usage)

in_token_q3 = result.usage.input_tokens
in_token_q5 = result_q5.usage.input_tokens

print(f"Initial Input Tokens: {in_token_q3}")
print(f"Chunked Input Tokens: {in_token_q5}")
print(f"Input Tokens Ratio Before vs After Chunked: {in_token_q3//in_token_q5}x")
```

#### **Answer**
Comparing 7126 tokens (full pages) to roughly 2309 tokens (chunks), the prompt is approximately **3 times smaller**.

The correct answer is **3× fewer**

## Q6. Turning it into an agent

So far search runs once, with the exact query. Let's make it agentic: give
the LLM a `search` tool and let it decide when (and what) to search. We
suggest [toyaikit](https://github.com/alexeygrigorev/toyaikit), the small
agent library from the module, but you can use anything you like - the OpenAI
Agents SDK, PydanticAI, LangChain, or a hand-written loop.

If you go with toyaikit:

```bash
uv add toyaikit
```

Create a `search` function that uses the chunk index. Give it a type hint and
a docstring - most frameworks read them to build the tool schema for you.

Build an agent with your `search` tool and run it (with toyaikit, the same way
as in the ToyAIKit lesson). Use these instructions for the agent (they nudge
it to search a few times):

> You're a course teaching assistant. Answer the student's question using the
> search tool. Make multiple searches with different keywords before answering.

Ask it:

> How does the agentic loop work, and how is it different from plain RAG?

The agent decides on its own when to search and when to answer. Count how many
times it called the `search` tool.

How many times did the agent call `search`?

> Note: the agent decides this itself, so it varies a little between runs -
> pick the closest option. We measured this with OpenAI `gpt-5.4-mini`; with a
> different model or provider the number may differ, so keep that in mind.

* 0
* 4 <-- The Answer
* 10
* 20

### Q6 Answer
To answer Question 6, we will transition from a fixed RAG pipeline to an **agentic system** where the LLM is given a search tool and decides how to use it to answer a complex query.

We can use the following code:
```python
# Import Library
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

# Define the search tool
search_call_count = {"count": 0}

def search(query: str) -> List[Dict]:
    """Search the course lesson chunks for relevant passages matching the query."""
    search_call_count["count"] += 1
    return chunk_index.search(query, num_results=5)

# Register the tool
tools = Tools()
tools.add_tool(search)
tools.get_tools()

# Build the Agent (toyaikit path)
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)
openai_client = OpenAIClient(model="gpt-5.4-mini", client=OpenAI())

instructions = """
You're a course teaching assistant. Answer the student's question using the search tool. 
Make multiple searches with different keywords before answering.
"""

runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=openai_client,
)

# Run the agentic query
query = "How does the agentic loop work, and how is it different from plain RAG?"
result = runner.loop(prompt=query, callback=callback)

# Read the counts
print(f"Total search calls: {search_call_count["count"]}")
```

#### **Answer**
The actual run resulting in 3 times of search call, so the nearest answer is **4**.

## **Notebook**
For the complete work, you can review my notebook: [hw_01.ipynb](..\01-agentic-rag\code\hw_01.ipynb) in the homework-01 [code folder](..\01-agentic-rag\code).

**Thank you.**