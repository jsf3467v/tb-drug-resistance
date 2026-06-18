# Deployment

Memgraph operates in Docker, while the application and evaluation scripts run on the host. Since this is a local portfolio prototype and not a hosted service, there is no cloud provisioning.

## Prerequisites

- Python 3.10, 3.11, or 3.12, the versions the test suite runs against.
- Docker.
- Git.

The `memgraph/memgraph-mage` image runs natively on Apple Silicon, so an M-series Mac needs no extra configuration.

## 1. Start Memgraph

Run the database in a container, detached so it stays up after the terminal closes. Port 7687 carries the Bolt connection the code uses, and port 7444 streams logs to Memgraph Lab. The tag pins a Memgraph release at or above 3.2, the version where a write inside a read transaction is rejected, which keeps the natural-language path read-only.

```bash
docker run -d -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage:3.9.0
```

If the container already exists from an earlier run, resume it rather than creating a new one.

```bash
docker start memgraph
```

To keep the data across container restarts, mount a named volume.

```bash
docker run -d -p 7687:7687 -p 7444:7444 -v mg_lib:/var/lib/memgraph --name memgraph memgraph/memgraph-mage:3.9.0
```

## 2. Memgraph Lab, optional

Memgraph Lab is the visual interface for browsing the graph. Install the desktop app, or run it in a container and open `localhost:3000`. On macOS the host address is `host.docker.internal`.

```bash
docker run -p 3000:3000 -e QUICK_CONNECT_MG_HOST=host.docker.internal memgraph/lab
```

## 3. Clone and install

```bash
git clone https://github.com/jsf3467v/tb-drug-resistance.git
cd tb-drug-resistance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Datasets

The real datasets are not stored in the repository because of their size. Download them into a `Datasets/` folder at the project root. The expected files are listed below.

- `Datasets/WHO-UCN-TB-2023.7-eng.xlsx`, the WHO mutation catalog.
- `Datasets/EFFECTS.parquet`, the catalog-graded mutations.
- `Datasets/PREDICTIONS.parquet`, the catalog genotypic calls.
- `Datasets/DST_MEASUREMENTS.parquet`, the measured DST phenotypes.
- `Datasets/UKMYC_PHENOTYPES.parquet`, the measured UKMYC phenotypes.
- `Datasets/DRUG_CODES.csv`, the three-letter drug code map.

The WHO catalog comes from the World Health Organization. The CRyPTIC tables come from the CRyPTIC consortium release on Zenodo. The synthetic patient cases are produced in code and need no download. Reading the parquet tables needs the pyarrow engine, which `requirements.txt` installs.

## 5. Environment

Copy the template and fill in the values.

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` for the natural-language interface. The graph variables default to the local no-auth instance, so `NEO4J_USER` and `NEO4J_PASSWORD` can stay empty and `NEO4J_URI` can stay at `bolt://localhost:7687`.

## 6. Build the graph

```bash
python SRC/tb_ontology.py
```

This clears the graph, applies the schema, loads the seed strains and patients, and merges the WHO catalog as 1,291 nodes from the 1,383 rows it grades 1 or 2, then prints `Database initialized successfully`. The seed strains and patients load whether or not the datasets are present, so the demo runs on the seed graph alone. The WHO catalog merge is skipped with a printed note when the catalog file is absent. The graded catalog reloads on every run, so a complete build is quick.

To reproduce the validation metrics instead, run `python Evaluation/validation.py`, which builds the graph and then scores the expert-system, case-based-reasoning, and CRyPTIC arms. The per-drug resistance scoring runs separately with `python Evaluation/metrics.py` and writes `per_drug_results.json`.

## 7. Run the application

```bash
streamlit run SRC/app.py
```

The front end opens in the browser. Enter your Anthropic API key in the sidebar, since the natural-language interface uses it to turn plain-English questions into graph queries. Click Initialize CBR once to load the 1,000 synthetic patient cases into the graph.

From there you can ask questions in plain English and query the synthetic data, for example to see which regimen a patient should receive. Each query returns four tabs, the direct answer, the expert-system rule trace, the case-based reasoning with its similar cases and success rate, and the generated Cypher. Every step is shown, so the reasoning behind a recommendation is auditable end to end.

## 8. Run the tests

```bash
pytest Evaluation/test_core.py
```

The suite needs no database, API, or datasets, so it runs immediately after the install step. The same suite runs in continuous integration on every push.

## Managing the container

```bash
docker stop memgraph     # stop without removing
docker start memgraph    # resume the same container
docker rm memgraph       # remove once stopped
```

## Troubleshooting

- If the application cannot reach the database, confirm the container maps port 7687 and that `NEO4J_URI` points to `bolt://localhost:7687`.
- If Memgraph fails to start, check the `vm.max_map_count` setting described in the Memgraph system configuration guide.
- If the expert-system arm of the validation is skipped, the API key is missing or unreachable. The CRyPTIC classification arm still runs and writes its results.