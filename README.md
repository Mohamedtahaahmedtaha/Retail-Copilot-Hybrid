# Retail Analytics Copilot (DSPy + LangGraph Hybrid Agent)

This project implements a local, free AI agent designed to answer retail analytics questions based on the Northwind Traders dataset. It combines **Retrieval-Augmented Generation (RAG)** over local policy and KPI documents with **SQL generation** against a local SQLite database. The agent uses **LangGraph** for workflow orchestration and **DSPy** for reliable, typed output generation.

---

## Setup and Execution

### Requirements

* Python 3.11+
* Git
* Ollama (must be installed and running)

### Model Configuration

[cite_start]The system is designed to use **Phi-3.5-mini-instruct** (3.8B)[cite: 11, 183]. However, due to system resource constraints encountered during development (RAM available was insufficient for the required 2.4 GB), the final submitted code is configured for the lower-resource **TinyLlama** model to ensure stability and completion.

### Running the Evaluation

To execute the batch run and generate the final `outputs_hybrid.jsonl` file:

```bash
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs/outputs_hybrid.jsonl
```
## Core Architecture: LangGraph Workflow
The agent utilizes a stateful LangGraph (AgentState) with three primary entry paths (rag, sql, hybrid) and a total of 7 core functional nodes and a dedicated Repair Loop.

State Management (AgentState)
The state dictionary manages context flow and error tracking:

question: The user's original query.

path: The path chosen by the Router (rag, sql, hybrid).

retrieved_docs: Chunks retrieved from the RAG system.

planner_output: Extracted constraints (dates, KPIs, categories).

sql_query: The generated SQLite query string.

sql_result: Results or error message from the database executor.

sql_error: Boolean flag to trigger the repair loop.

Node	Module Type	Description
1. Router	DSPy Classifier	Determines the required path: rag (policy/definitions), sql (simple numerical), or hybrid (RAG + SQL logic).
2. Retriever	Custom RAG	Uses a simple approach (e.g., TF-IDF or BM25) over chunked markdown files (docs/) to return top-K context chunks and their unique IDs.
3. Planner	DSPy Predict	(Hybrid Path) Extracts constraints (dates from documents, KPI definitions, categories) necessary for SQL generation from retrieved context.
4. NL-SQL	DSPy Predict	Generates a valid SQLite query using the live schema (obtained via PRAGMA) and the constraints from the Planner.
5. Executor	Tool/Function	Executes the generated SQL against northwind.sqlite and captures the columns, rows, or any database error.
6. Synthesizer	DSPy Predict	(Optimized) Generates the final answer, strictly matching the format_hint, and ensures full citation completeness.
7. Check / Repair	Conditional Logic	Required Repair Loop: Checks for sql_error or invalid output format. If an error is found and repair_count < 2, it reroutes to a repair state; otherwise, it terminates.

repair_count: Counter for the repair loop attempts.

Graph Nodes and Flow

طك
