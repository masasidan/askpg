# AskPG

Paul Graham's writing has answered a lot of my questions. It has also left me with
many more.
AskPG grew out of a simple idea: turn the body of work he has already made public into
a conversation.

It is a terminal chat backed by Paul Graham's public essays and authored tweets. It is
not fine-tuned on him. Instead, it uses retrieval-augmented generation (RAG) to find
relevant things he has written and gives those passages to an OpenAI model before it
answers.

AskPG is not Paul Graham, is not affiliated with him, and does not claim his
endorsement. The terminal stays immersive during normal conversation, but `/sources`
lets you inspect the evidence behind the most recent answer.

## What it does

- Scrapes the canonical essay index at [paulgraham.com](https://paulgraham.com/articles.html).
- Imports a broad public archive of authored tweets spanning 2010–2026.
- Splits essays at sections and paragraph-level idea boundaries.
- Combines SQLite full-text search with OpenAI semantic embeddings.
- Rewrites conversational follow-ups into standalone searches and reranks the results.
- Keeps persistent conversation memory across terminal sessions.
- Accepts screenshots and other image attachments for visual questions.
- Answers in an immersive first-person voice while guarding against character breaks.
- Hides citations by default; `/research` and `/sources` make the evidence visible.

## Quick start

```bash
git clone https://github.com/masasidan/askpg.git
cd askpg

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

cp .env.example .env
# Open .env and replace the placeholder with your OpenAI API key.

askpg scrape
askpg scrape-tweets
askpg index
./chat
```

The first full index build creates semantic embeddings for the downloaded corpus and
therefore uses the OpenAI API. Existing embeddings are stored locally and reused. To
build only the free local full-text index, run `askpg index --skip-embeddings`.

## Chat commands

| Command | Action |
| --- | --- |
| `/sources` | Show evidence used for the last answer |
| `/research` | Turn inline citations and source lists on |
| `/immersive` | Hide citations and source lists again |
| `/memory` | Show how many conversation turns are stored locally |
| `/clear` | Permanently clear conversation memory |
| `/quit` | Exit |

## Other commands

```text
askpg scrape                             refresh essays from paulgraham.com
askpg scrape-tweets                      refresh the historical tweet archive
askpg index                              sync the corpus and add missing embeddings
askpg index --skip-embeddings            build only the SQLite full-text index
askpg embed                              add missing semantic embeddings later
askpg search "how to find ideas"         inspect retrieved passages
askpg ask "How should I pick an idea?"   ask one immersive question
askpg ask "What is wrong here?" --image screenshot.png
askpg ask --research "..."               ask with citations and sources visible
askpg chat                               start an interactive conversation
askpg stats                              show corpus and embedding statistics
askpg doctor                             check setup without exposing your key
```

## How the RAG pipeline works

1. A follow-up question is rewritten into a standalone search.
2. Full-text and semantic search retrieve candidates from essays and tweets.
3. A smaller model reranks those candidates for direct relevance.
4. The seven best passages, recent dialogue, and relevant older memories are sent to
   the answer model.
5. The answer model responds; immersive mode removes citation markers from the display.

Recent dialogue is bounded rather than sending every past conversation. Older messages
are searched locally and included only when relevant.

On macOS, copy a screenshot and press Ctrl+V while the `You:` prompt is active. AskPG
shows `[attach 1]` beside the prompt without inserting it into your message. You can
paste more than one image. With the text input empty, Backspace removes the most recent
attachment. Command+V continues to paste ordinary text through the Terminal. Attached
images are sent only with the next question and are not added to persistent memory.

## Models and API usage

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | none | OpenAI authentication |
| `ASKPG_MODEL` | `gpt-5.6-terra` | Final answer model |
| `ASKPG_REASONING_EFFORT` | `medium` | Answer reasoning effort |
| `ASKPG_RETRIEVAL_MODEL` | `gpt-5.6-luna` | Query rewriting and reranking |
| `ASKPG_EMBEDDING_MODEL` | `text-embedding-3-small` | Semantic embeddings |
| `ASKPG_DATA_DIR` | `./data` | Corpus, memory, and index location |

A normal semantic question may use four API operations: query rewriting, query
embedding, passage reranking, and final generation. Document embeddings are cached
locally and are not regenerated unless the corpus or embedding model changes.

## Data, privacy, and source material

The repository intentionally contains no API key, scraped essays, tweets, embeddings,
or conversation history. These are created under the git-ignored `data/` directory on
your machine.

Responses are sent with `store=False`. The local corpus and full conversation database
remain on your machine, but the query context and selected passages needed for an
answer are sent to the OpenAI API. Attached images are also sent to the API for visual
analysis, but AskPG does not copy them into its data directory.

The tweet importer uses the public
[`aaahmet/paulg-tweets`](https://huggingface.co/datasets/aaahmet/paulg-tweets) archive
and retains original tweet IDs and X URLs. Essays and tweets remain their authors'
copyrighted material. Keep downloaded source material for personal research, preserve
attribution, and review applicable website and platform terms before redistribution.

## Testing

```bash
python -m unittest discover -s tests -v
```
