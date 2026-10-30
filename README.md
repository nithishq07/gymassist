# GymAssist – RAG-Based Fitness Assistant

NLP AAT project: a small Retrieval-Augmented Generation (RAG) assistant that answers gym and exercise questions from a 207-exercise knowledge base.

> **Credit.** Based on the open-source "fitness-assistant" demo project (LLM Zoomcamp). The original RAG demo is not my work. My contribution is the customisation (Streamlit-only interface, cleaned-up code), a baseline vs improvement comparison (TF-IDF vs dense retrieval) and the evaluation.
> `[TODO: student to add the original repository URL and check its licence – the zip I received had no LICENSE file.]`

## Current status (read this first)

| Part | Status |
|---|---|
| TF-IDF retrieval metrics (25 draft test questions + 200 LLM-generated questions) | **Run for real** (no OpenAI needed). Test labels are still DRAFT, so treat the numbers as provisional. |
| Dense retrieval metrics | **PENDING** – needs `OPENAI_API_KEY`; run `python evaluation.py` |
| LLM answer relevance (RELEVANT / PARTLY / NON) | **PENDING** – run `python evaluation.py --judge` |
| Example answers in section I | **PENDING** – need an OpenAI run |
| Test-question ids | **DRAFT – student must verify every id manually** (`data/test_questions.csv`) |

Numbers in this README come from `results/results.json`. Nothing here was copied from the original project's README except where it says "reported by the original project".

## How to run

```bash
pip install -r requirements.txt

# set your key (do not commit it)
export OPENAI_API_KEY="sk-..."          # macOS / Linux
# $env:OPENAI_API_KEY="sk-..."          # Windows PowerShell

python dense_retriever.py               # once: builds data/embeddings.npy (207 vectors)
streamlit run app.py                    # the demo app
python evaluation.py                    # retrieval metrics -> results/results.json
python evaluation.py --judge            # also LLM-judge answer relevance (about 125 OpenAI calls)
python preprocessing.py                 # re-creates data.csv from data_unclean.csv and compares
python baseline.py "Which muscles do push-ups work?"   # TF-IDF baseline, no LLM
```

The app needs `OPENAI_API_KEY` for every question (the answer is written by gpt-4o-mini). The dense option additionally needs `data/embeddings.npy` (or builds it on first use).

## Project structure

```text
app.py              Streamlit UI (the only interface)
rag_pipeline.py     retrieve top-k -> build prompt -> LLM; method = tfidf | dense
preprocessing.py    data_unclean.csv -> data.csv (and checks it matches)
baseline.py         TF-IDF + cosine only, no boosting, no LLM
dense_retriever.py  in-memory dense retrieval (OpenAI embeddings + NumPy cosine)
evaluation.py       Hit Rate@k, Recall@k, MRR; optional LLM judge
minsearch.py        TF-IDF search index from the original project (unchanged)
data/               data.csv, data_unclean.csv, test_questions.csv, original CSVs
docs/               literature_review.md
results/            results.json
original_unused/    original Flask API, CLI, Postgres, Grafana, Docker, notebooks (kept, not used)
```

## About the project

- **Problem:** fitness users need quick, relevant answers about exercises; a general chatbot may not ground its answers in a specific knowledge source.
- **Objective:** a simple RAG fitness assistant that retrieves from a fitness knowledge base and generates a context-aware answer with an LLM.
- **Technologies actually used:** Python, Streamlit, pandas, NumPy, scikit-learn (TF-IDF and cosine similarity), OpenAI API (gpt-4o-mini for answers, text-embedding-3-small for dense retrieval). No LangChain, no vector database, no chunking.
- **How RAG works here:** question → (TF-IDF vector or embedding) → search the knowledge base → top-k exercises → context + question go into the LLM prompt → answer.
- **Why RAG is an NLP technique:** RAG combines natural-language question answering (the user asks in ordinary words), information retrieval (finding the exercise records that match the words or meaning of the question) and language-model generation (writing a natural-language answer from the retrieved text). All three work on human language.

## AAT sections

### A. NLP topic
Natural Language Question Answering, Information Retrieval and Retrieval-Augmented Generation (RAG).

### B. Application
A gym/fitness question-answering assistant (exercise selection, alternatives, muscles worked, equipment, instructions).

### C. Dataset
`data/data.csv` – **207 exercises, 8 columns**: `id` (0–206, unique), `exercise_name`, `type_of_activity`, `type_of_equipment`, `body_part`, `type`, `muscle_groups_activated`, `instructions`. No missing values and no repeated exercise names.

| Column | Distinct values | Most common values |
|---|---|---|
| type_of_activity | 7 | Strength (182), Cardio/Strength (13), Strength/Mobility (4), Cardio (4), Mobility (2), Cardio/Warm-up (1), Strength/Cardio (1) |
| type_of_equipment | 20 | Bodyweight (60), Dumbbells (35), Cable Machine (26), Barbell (21), Machine (21), Kettlebell (11), … (note: "Dumbbell" 5 and "Dumbbells" 35 are separate values) |
| body_part | 4 | Upper Body (95), Lower Body (61), Core (34), Full Body (17) |
| type | 6 | Push (110), Pull (60), Hold (28), Twist (7), Stretch (1), Push/Pull (1) |

- **Provenance:** bundled with the open-source fitness-assistant project. The original README's file listing describes `data.csv` as "generated with ChatGPT"; no other collection details are documented in the repo. `[TODO: student to confirm source]`
- **Suitability:** each row is a structured exercise record (muscles, equipment, body part, written instructions), which suits question answering about exercises. It only contains exercises, so it cannot answer conceptual questions (see limitations).
- `data/data_unclean.csv` is the raw semicolon-delimited version (209 rows, capitalised headers, no id).
- `data/ground-truth-retrieval.csv` (1,035 LLM-generated questions, 5 per exercise) and `data/rag-eval-*.csv` are from the original project, not mine.

### D. Preprocessing
`preprocessing.py` does exactly this: read the raw CSV (`;` delimiter) → headers to snake_case → strip whitespace → drop exact duplicate rows → add `id`.

I ran it and compared with `data/data.csv`: **MATCH** (the output is identical, cell for cell and byte for byte). What it actually did on this data: 209 raw rows → 0 cells changed by whitespace stripping → 2 exact duplicate rows dropped (Ab Wheel Rollout and Smith Machine Incline Press appeared twice) → 207 rows.

**Real pipeline from text to retrieval:**
1. raw CSV → cleaning → `data.csv` (207 rows)
2. TF-IDF path: each text field gets its own TF-IDF vectoriser (scikit-learn defaults: lower-casing, no stop-word removal, no stemming); scores of the 7 fields are added up with optional boosts.
3. Dense path: each row is written as one text document with `entry_template` (the same text format that goes into the LLM prompt) and embedded with text-embedding-3-small.
4. **No chunking is needed** because one exercise = one short document.

### E. Baseline
`baseline.py`: TF-IDF + cosine similarity over the exercise fields, **no boosting, no LLM**. The "answer" is just the top retrieved exercise. It reuses `minsearch.py` from the original project.

### F. Proposed method (RAG) and the improvement

```text
question
   │
   ├── tfidf: TF-IDF vector (original boosts)  ─┐
   └── dense: question embedding               ─┤
                                                ▼
                    search knowledge base (207 exercises) → top-10 exercises
                                                ▼
                    prompt = instructions + QUESTION + CONTEXT (the 10 exercises)
                                                ▼
                    gpt-4o-mini → answer shown in Streamlit (with the retrieved exercises)
```

**Improvement (the one real addition):** `dense_retriever.py` embeds the 207 documents with OpenAI `text-embedding-3-small`, caches the vectors in `data/embeddings.npy`, and searches with NumPy cosine similarity. This is *in-memory dense retrieval*, not a vector database. It is wired into `rag_pipeline.py` as an alternative to TF-IDF (the "Retrieval method" selector in the app).

Other small changes to the original logic: fixed the "insrtuctor" typo in the prompt, the OpenAI client is created only when needed, and the LLM judge is **not** run on normal app questions.

### G. Evaluation
**Metrics** (computed on the top-10 list of each retriever; a question has one or more relevant exercise ids):
- **Hit Rate@k:** share of questions where at least one relevant exercise is in the top k.
- **Recall@k:** share of the relevant exercises that are in the top k (averaged over questions).
- **MRR@10:** average of 1/rank of the first relevant exercise (0 if none in the top 10).

**Test sets**
- `data/test_questions.csv`: 25 questions (10 easy, 8 paraphrased, 7 hard). **DRAFT – student must verify every id manually.** I drafted the ids by searching `data.csv`; some may be incomplete or debatable.
- Secondary check: a random sample of 200 of the original project's 1,035 LLM-generated questions (seed 42). **LLM-generated, not manually verified.** One relevant id each, so Hit@k = Recall@k.

**Retrieval results – 25 draft test questions (provisional until ids are verified)**

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| TF-IDF baseline (no boosting) | 0.560 | 0.680 | 0.720 | 0.760 | 0.599 | 0.651 | 0.635 |
| TF-IDF with original boosts | 0.560 | 0.680 | 0.720 | 0.720 | 0.609 | 0.670 | 0.630 |
| Dense (text-embedding-3-small) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

By question type (Hit@10 and MRR@10):

| Question type (n) | TF-IDF baseline Hit@10 | TF-IDF baseline MRR@10 | Dense Hit@10 | Dense MRR@10 |
|---|---|---|---|---|
| easy (10) | 1.000 | 0.900 | PENDING | PENDING |
| paraphrased (8) | 0.500 | 0.328 | PENDING | PENDING |
| hard (7) | 0.714 | 0.607 | PENDING | PENDING |

**Retrieval results – 200 LLM-generated questions (not manually verified)**

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| TF-IDF baseline (no boosting) | 0.765 | 0.820 | 0.860 | 0.930 | 0.860 | 0.930 | 0.809 |
| TF-IDF with original boosts | 0.855 | 0.895 | 0.905 | 0.930 | 0.905 | 0.930 | 0.879 |
| Dense (text-embedding-3-small) | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

**LLM-judge answer relevance (25 test questions, TF-IDF-only answer vs RAG answers):** PENDING – run `python evaluation.py --judge`.
Note that the judge (from the original project) sees only the question and the answer, not the retrieved context, so it measures relevance, not whether the answer is faithful to the context.

**Reported by the original project (not my results):** on its 1,035 generated questions it reports TF-IDF without boosting at hit rate 94% and MRR 82%, and with tuned boosts hit rate 94% and MRR 90%; its LLM judge on 200 sampled questions labelled 167 answers RELEVANT, 30 PARTLY_RELEVANT and 3 NON_RELEVANT for gpt-4o-mini.

### H. Comparison table

| | TF-IDF baseline | Dense retrieval | RAG generation layer (on top of either retriever) |
|---|---|---|---|
| Lexical matching | Yes – word overlap, equal field weights | No – compares embedding vectors, not words | Depends on the retriever chosen (yes with TF-IDF, no with dense) |
| Semantic understanding | No | Yes, expected from embeddings – to be confirmed by the PENDING dense results | Only through the retriever and the LLM reading the retrieved text |
| Uses retrieved context | Yes – returns it directly | Yes | Yes – the top-10 exercises are put in the prompt |
| LLM-generated answer | No | No (retrieval only) | Yes – gpt-4o-mini |
| Handles paraphrased natural-language queries | Limited: on my 8 draft paraphrased questions TF-IDF found a relevant exercise in the top 10 for 4 (baseline) / 3 (boosted) of them | PENDING (not measured yet) | PENDING (not measured yet) |

### I. Results and discussion

**What the real TF-IDF runs show (draft labels, n = 25, so only indicative):**
- Easy questions that contain the exercise name or equipment are retrieved well (baseline Hit@3 = 1.000 on the 10 easy questions).
- Paraphrased questions are where TF-IDF is weakest (baseline Hit@10 = 0.500, boosted Hit@10 = 0.375 on 8 questions). This is the gap the dense retriever is meant to test; **it is not yet measured**.
- On the 25 draft questions the original boosts did not help (Hit@10 0.720 vs 0.760 without boosts, MRR 0.630 vs 0.635). On the 200 generated questions they did help (MRR@10 0.879 vs 0.809), but the boosts were tuned by the original author on questions of that kind, so that is not a fair test of the boosts.
- With only 25 questions one question changes a score by 0.04, so small differences mean little.

**Examples** (retrieved exercises are from a real TF-IDF run; answers need OpenAI, so they are PENDING):

| # | Question | TF-IDF top-5 retrieved (real run) | Final RAG answer | Relevant? |
|---|---|---|---|---|
| 1 | Give me a beginner chest workout. | Chest Press, Cable Chest Press, Machine Chest Press, Machine Chest Fly, Incline Chest Press | PENDING – needs an OpenAI run | PENDING |
| 2 | Which muscles do push-ups work? | Push-Ups, TRX Push-Ups, Incline Push-Ups, Decline Push-Ups, Spider Push-Ups | PENDING – needs an OpenAI run | PENDING |
| 3 | Suggest an alternative to squats. | Squats, Goblet Squats, Push-Up to Row, Jumping Squats, Pistol Squat to Box | PENDING – needs an OpenAI run | PENDING |
| 4 | Is there a stretch for the back of my thighs? | Machine Back Extension, Hamstring Stretch, Romanian Deadlift, Deadlift, Bulgarian Deadlift | PENDING – needs an OpenAI run | PENDING |
| 5 | What is progressive overload? **(failure case)** | One-Legged Squat, Pistol Squat, Single-Leg Squat, Bulgarian Split Squats, Cable Lateral Raise | PENDING – needs an OpenAI run | PENDING |

The failure case "What is progressive overload?" is kept out of the demo on purpose. No row of `data.csv` contains "progressive" or "overload", so the knowledge base cannot answer it. TF-IDF still returns unrelated exercises (listed above), and with dense retrieval the system would always return the 10 nearest exercises, because there is no similarity threshold. What the LLM does with that context is PENDING.

**Limitations**
- Only 207 exercises, and no difficulty or "beginner" field (the word "beginner" appears in 0 rows), so "beginner" questions depend on the LLM's own judgement.
- No concept explanations (training principles, nutrition, injuries); the answer prompt does not tell the LLM to say "I don't know" when the context is missing.
- Answers depend on the LLM (gpt-4o-mini) and are not deterministic.
- Only the top 10 exercises are given to the LLM, so "list everything with dumbbells" (41 exercises have a dumbbell value) cannot be complete.
- TF-IDF breaks ties between equal scores in an arbitrary order, and "Dumbbell" and "Dumbbells" are different words for it.
- The test set is small (25) and its labels are drafts; the 200-question sample is LLM-generated.
- Not medical or personal-training advice.

### J. Literature review
See [`docs/literature_review.md`](docs/literature_review.md) – five papers (RAG, DPR, Sentence-BERT, BEIR, RAG survey). Verify each citation before submission.

### K. Research gap
Keyword retrieval can struggle with differently worded fitness questions. This project tests whether semantic (dense) retrieval plus RAG gives more relevant, grounded answers than TF-IDF on a small, domain-specific exercise knowledge base. The BEIR paper warns that dense retrieval does not always beat lexical methods, so the result is not assumed: the dense numbers are PENDING and will be reported whichever way they turn out.
