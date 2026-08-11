# Deployment

Memgraph runs in Docker while the application and evaluation scripts run on the host. This is a local portfolio prototype rather than a hosted service, so there is no cloud provisioning.

## Prerequisites

- Python 3.10, 3.11, or 3.12, the versions the test suite runs against
- Docker
- Git

The version floors in `requirements.txt` are the oldest release of each dependency that installs and runs across all three Python versions, verified by building an environment at every floor and running the suite. Two are set by a working install rather than by an available package. Older pyarrow releases ship no wheel for Python 3.12, and anthropic releases before 0.40.0 install cleanly and then fail when the client is constructed, because they pass an argument that httpx removed.

The `memgraph/memgraph-mage` image runs natively on Apple Silicon, so an M-series Mac needs no extra configuration.

## 1. Start Memgraph

Run the database detached so it survives the terminal closing. Port 7687 carries the Bolt connection the code uses and port 7444 streams logs to Memgraph Lab. The tag pins a release at or above 3.2, the version where a write inside a read transaction is rejected, which is what keeps the natural-language path read-only.

```bash
docker run -d -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage:3.9.0
```

If the container already exists from an earlier run, resume it rather than creating another.

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
pip install -r requirements-dev.txt
```

`requirements.txt` carries runtime dependencies only. `requirements-dev.txt` pulls that file in and adds `pytest`, so the single command above covers both running the application and running the test suite. Install `requirements.txt` alone if you do not intend to run the tests.

## 4. Datasets

The real datasets are not stored in the repository because of their size. Download them into a `Datasets/` folder at the project root.

- `Datasets/WHO-UCN-TB-2023.7-eng.xlsx`, the WHO mutation catalog
- `Datasets/EFFECTS.parquet`, the catalog-graded mutations
- `Datasets/PREDICTIONS.parquet`, the catalog genotypic calls
- `Datasets/DST_MEASUREMENTS.parquet`, the measured DST phenotypes
- `Datasets/UKMYC_PHENOTYPES.parquet`, the measured UKMYC phenotypes
- `Datasets/DRUG_CODES.csv`, the three-letter drug code map

Some copies of the WHO release separate the version with an underscore rather than a dot. The reader matches either spelling, so the file needs no renaming.

The release also ships `DATA_SCHEMA.pdf` and `MUTATIONS.parquet`. Neither is read, so neither needs downloading, which keeps the download near forty megabytes rather than a gigabyte.

The WHO catalog comes from the World Health Organization and the CRyPTIC tables from the CRyPTIC consortium release on Zenodo. The synthetic patient cases are produced in code and need no download. Reading the parquet tables needs the pyarrow engine, which `requirements.txt` installs.

A seventh file, `Datasets/cryptic_features.parquet`, is not downloaded. The first run that needs it builds it from `DST_MEASUREMENTS.parquet`, `UKMYC_PHENOTYPES.parquet`, `PREDICTIONS.parquet`, and `DRUG_CODES.csv`, then reuses that copy on every later run. The cache carries its own invalidation, rebuilding whenever any of those four, or `feature_engineering.py`, or `config.py` is newer than the cached file, so replacing a source table is enough and no manual delete is needed. `EFFECTS.parquet` and the WHO workbook are read directly and are not cached at all.

Rebuild explicitly to print the isolate count, the label balance, and the DST and UKMYC concordance the README reports.

```bash
python SRC/feature_engineering.py
```

## 5. Environment

Copy the template and fill in the values.

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` for the natural-language interface. The graph variables default to the local instance with no authentication, so `NEO4J_USER` and `NEO4J_PASSWORD` can stay empty and `NEO4J_URI` can stay at `bolt://localhost:7687`.

## 6. Build the graph and score the system

Two paths lead here, and which one you take decides whether section 6.1 is needed at all.

### 6.1 Demo only

Build the graph directly when you want the application without the scores.

```bash
python SRC/tb_ontology.py
```

This clears the graph, applies the schema, loads the seed strains and patients, merges the WHO catalog as 1,291 nodes from the 1,383 rows it grades 1 or 2, and prints `Database initialized successfully`. The seed strains and patients load whether or not the datasets are present, so the demo runs on the seed graph alone, and the catalog merge is skipped with a printed note when the workbook is absent. The graded catalog reloads on every run, so a complete build stays quick.

### 6.2 Full run

`validation.py` rebuilds the graph itself, performing the same clear, schema, seed, and catalog merge, so running section 6.1 first is work it immediately discards. Take the order below instead, which puts the cheap checks ahead of the step that spends money.

```bash
python SRC/who_catalog.py           # the catalog parses, writes nothing
python SRC/cbr_cases.py             # the generator runs, writes nothing
python SRC/feature_engineering.py   # rebuilds the label cache
python -m pytest tests/ -q          # 127 tests, no database and no API
python Evaluation/metrics.py        # per-drug scores, no database and no API
python Evaluation/validation.py     # rebuilds the graph, calls the API
```

The first two write nothing and exist to fail early. The next two need neither the database nor the API, so a failure in either is a data problem rather than an infrastructure one. Only the last step rebuilds the graph and spends API calls.

Run `python Evaluation/validation.py --fresh` the first time, or after editing the schema or the prompt examples. The expert arm journals its progress so an interrupted run can resume, and the journal carries a digest of the model together with the schema and the examples, so editing any of them discards results produced under the old prompt rather than resuming on top of them. A journal file left on disk means the last run did not finish.

Because section 6.2 clears the graph, run it before section 7 rather than after. It discards anything the running application loaded, including the case base.

Both scoring scripts resolve their output against their own location rather than the working directory, so they write `Evaluation/validation_results.json` and `Evaluation/per_drug_results.json` no matter where you launch them from, replacing the committed reference copies. The validation report merges rather than overwrites, so an arm that was skipped keeps its previous result, and an `arms_this_run` field records which sections were recomputed. Back up the two files before a rerun if you want the originals kept. The expert-system arm calls a live model and is the one figure that moves between runs.

## 7. Run the application

```bash
streamlit run SRC/app.py
```

The front end opens in the browser. Enter your Anthropic API key in the sidebar, since the natural-language interface uses it to turn plain-English questions into graph queries. Click Initialize CBR once to load the 1,000 synthetic patient cases into the graph.

From there you can ask questions in plain English and query the synthetic data, for example to see which regimen a patient should receive. Each query returns four tabs, the direct answer, the expert-system rule trace, the case-based reasoning with its similar cases and success rate, and the generated Cypher. Every step is shown, so the reasoning behind a recommendation is auditable end to end.

## 8. Run the tests

```bash
python -m pytest tests/ -q
```

The 127 tests need no database, API, or datasets, so they run immediately after the install step, provided `pytest` was installed there. The suite runs from the project root or from inside `tests/`. The same suite runs in continuous integration on every push to main and on every pull request, across Python 3.10, 3.11, and 3.12.

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
- If the scores do not move after you replace a dataset, check that the replacement carries a newer modification time than `Datasets/cryptic_features.parquet`. Copying a file can preserve the original timestamp, which is the one case the cache cannot see. Delete the cache and run again.
- If `pytest` is not found, install it as shown in section 3. It is not a runtime dependency.
- If the expert arm re-queries when you expected it to resume, the schema or the prompt examples changed since the journal was written, so the results produced under the old prompt were discarded rather than mixed with new ones.
- If the application reports no cases after a scoring run, section 6.2 cleared the graph. Click Initialize CBR again to reload them.
