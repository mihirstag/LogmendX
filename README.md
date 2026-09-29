# LogMend 2.0

LogMend is a research prototype for investigating endpoint process logs. It combines a local large language model (LLM) planner with anomaly scoring, signature matching, process-graph context, MITRE ATT&CK mapping, and threat-intelligence lookup. Investigations run as a bounded Thought–Action–Observation (TAO) loop and produce a severity assessment and a structured analyst report.

This repository contains a LogMend implementation related to the LogRESP-Agent research paper cited below. It is not a reproduction of the paper's exact implementation: this code routes planning through Ollama and uses the model ensembles and tools described here. Results reported in the paper should not be treated as benchmark results for this repository.

## Capabilities

- Parses flat and nested BETH process-event records and UUID-keyed DARPA THEIA records.
- Dynamically selects among six investigation tools: `DescriptionGenerator`, `RuleMatcher`, `SequenceScorer`, `ContextRetriever`, `TTPMapper`, and `ThreatLookup`.
- Scores BETH processes with GraphCL, Isolation Forest, and Deep SAD; scores DARPA processes with MAGIC, FLASH, and ORTHRUS.
- Uses Neo4j provenance-graph context and sentence-embedding similarity to map behavior to a small MITRE ATT&CK knowledge base.
- Stops investigations when confidence thresholds are met, tools are exhausted, or the tool-call limit is reached.
- Reports a verdict, severity, evidence, reasoning trace, and recommended analyst action. The prototype does not automatically contain or remediate threats.

## How It Works

1. `LogParser` normalizes the input record and identifies its dataset.
2. The Ollama-backed planner proposes a hypothesis and selects an available tool.
3. The selected tool returns evidence, which is added to the investigation history.
4. The planner repeats until a stop condition is reached; LogMend then calculates severity and generates a report.

The scorer expects trained artifacts under `models/saved_models/`. Neo4j must contain the relevant process data for graph-backed investigation. BETH batch evaluation temporarily loads the selected JSON data into Neo4j before running the agent.

## Requirements

- Python 3.10 or newer.
- Ollama running locally or at a configured URL, with the selected model available.
- Neo4j with the BETH and/or DARPA provenance graph loaded for graph-backed investigations.
- Compatible PyTorch and PyTorch Geometric installations, plus the model artifacts expected under `models/saved_models/`.
- Python packages used by the application include `requests`, `python-dotenv`, `neo4j`, `numpy`, `scikit-learn`, `joblib`, and `sentence-transformers`. Install PyTorch and PyTorch Geometric using versions compatible with your Python and hardware. This repository currently has no pinned dependency manifest.

## Configuration

Create a local `.env` file in the project root. Do not commit credentials. Set the Ollama model to a model installed in your Ollama instance and configure the Neo4j credentials for your environment:

```dotenv
OLLAMA_URL=http://localhost:11434
LOGRESP_MODEL=qwen3.8:27b

NEO4J_BETH_URI=bolt://localhost:7687
NEO4J_BETH_USER=neo4j
NEO4J_BETH_PASS=replace-with-your-password

NEO4J_THEIA_URI=bolt://localhost:7687
NEO4J_THEIA_USER=neo4j
NEO4J_THEIA_PASS=replace-with-your-password
```

The model value above is the code's default; change it if that model is not installed. `json_file_runner.py` uses a local Neo4j connection and `NEO4J_BETH_PASS` for its temporary graph. Before running it, ensure those connection settings match your Neo4j instance and use a database suitable for test data. The runner removes its tagged temporary graph data when each run finishes.

## Run

From the repository root, run the built-in BETH and DARPA example investigations:

```powershell
python main.py
```

Provide your own authorized BETH JSON data locally; datasets are not included in the upload. Run a limited evaluation with a JSON file:

```powershell
python json_file_runner.py --file path/to/your/part_N.json --limit 10
```

Evaluate every JSON file in a directory:

```powershell
python json_file_runner.py --dir path/to/your/beth_json_directory --limit 10
```

Omit `--limit` to process all process groups in the selected input. These runs call Ollama and Neo4j and require the scorer's model artifacts; they are not offline-only commands. The runner writes a timestamped evaluation report in the working directory.

## Training

`train_beth_models.py` and `train_darpa_models.py` train the corresponding scoring models from Neo4j graph data and save artifacts under `models/saved_models/`. Configure and populate the appropriate Neo4j database before running them:

```powershell
python train_beth_models.py
python train_darpa_models.py
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main.py` | TAO investigation agent and report generation |
| `agent/ollama_router.py` | Ollama chat API client |
| `tools/` | Parsing, scoring, rules, context retrieval, TTP mapping, and threat lookup |
| `database/neo4j_router.py` | Neo4j connections for BETH and DARPA data |
| `models/architectures.py` | PyTorch Geometric model architectures |
| `models/saved_models/` | Trained model weights and scalers |
| `beth_test_data/` | BETH JSON event files for batch evaluation |
| `train_*_models.py` | BETH and DARPA model training scripts |
| `json_file_runner.py` | BETH JSON batch evaluation runner |
| `evaluate_metrics.py` | Evaluation metric utilities |

## Research Reference

Juyoung Lee, Yeonsu Jeong, Taehyun Han, and Taejin Lee. “LogRESP-Agent: A Recursive AI Framework for Context-Aware Log Anomaly Detection and TTP Analysis.” *Applied Sciences*, 15(13), 7237, 2025. [https://doi.org/10.3390/app15137237](https://doi.org/10.3390/app15137237). The article is available from the publisher at the DOI above.

The paper reports its own experimental results, including 99.97% accuracy and 97.00% F1 for binary detection and 99.54% accuracy and 99.47% F1 for multi-class classification. These are the paper's reported results, not independently verified results for this codebase. The article is published under the Creative Commons Attribution 4.0 license; see the paper for license terms.