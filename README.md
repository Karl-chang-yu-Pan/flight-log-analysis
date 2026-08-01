# PX4 Flight Log Analysis Agent — analysis-v2

This experimental branch tests a new analysis architecture while preserving the
existing upload, browse, and review frontend. The active analysis path uses one
autonomous OpenAI Agents SDK agent that can inspect a ULog with `pyulog`, write
and run its own Python checks, read the exact PX4 source snapshot, generate
plots, and return the existing structured report schema.

This is an internal-development project. It is not a hardened service for
untrusted files, prompts, or public deployment.

## What Changed on This Branch

- The web frontend now runs `flight_log_agent.analysis.analyzer` rather than the
  older mechanism-first DAG orchestration.
- The analyzer resolves the PX4 Git hash recorded in the ULog and reads that
  commit directly from Git objects. It does not check out or copy source trees.
- A single model context performs the investigation, writes Python verification
  scripts, tests alternatives, and decides when evidence is sufficient.
- Web search is available by default for external-context gaps and can be
  disabled for direct CLI or Python API runs.
- The default analysis limit is 30 turns with high reasoning effort.
- Questions and answers are stored as durable history and displayed as a
  timeline in the existing review UI.
- Flight Review-style plots and diagnostics remain available in the right
  sidebar, including ESC RPM when the log contains usable `esc_status` RPM
  fields.
- The Browse page can synchronize a Flight Review SQLite database into this
  application's own browse database without copying ULogs or modifying the
  source database.

The older `runner.py`, `flight_log_agent/runner_core.py`, and mechanism/DAG
modules remain in the repository for compatibility and tests. They are not the
active web analysis path on this branch, and `runner.py` is not the v2 CLI.

## Architecture

```text
ULog + question + optional mission
                │
                ▼
       deterministic pyulog prepass
       inventory + logged PX4 hash
                │
                ▼
     resolve hash to a full SourceSnapshot
                │
                ▼
       one autonomous analysis agent
       ├── sandboxed Python in work/
       ├── read-only ULog and mission
       ├── commit-pinned PX4 Git reads
       └── optional hosted web search
                │
                ▼
       structural report validation
                │
                ▼
       report.json + plots + audit logs
```

The report validator is a structural confidence guard, not a second factual
analyzer. Every hypothesis must describe an expected logged signature.
Medium- and high-confidence hypotheses must also contain source references and
numeric checks and must not claim required signals that are missing. Invalid
confidence is downgraded when necessary.
It does not independently verify the cited source or recalculate the model's
measurements.

## Requirements

- Linux
- Python 3.10 or newer
- Git
- `bubblewrap` (`bwrap`)
- ripgrep (`rg`)
- `sed` and `find`
- An OpenAI API key and access to the configured model for analysis runs

Bubblewrap, Git, ripgrep, `sed`, and `find` are host tools and should be
installed with the operating system's package manager. Python packages must be
installed in the repository virtual environment.

## Setup

Initialize the PX4 reference repository and create the virtual environment:

```bash
git submodule update --init --recursive

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

The PX4 Git repository must contain the firmware commit recorded in each ULog
you want to analyze. If a required commit or nested submodule commit is missing,
fetch it into the corresponding repository before running the analysis.

Create a local `.env` file for the web launcher:

```dotenv
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5.6

# Recommended unless you intentionally use OpenAI Agents SDK tracing:
OPENAI_AGENTS_DISABLE_TRACING=true
# Alternative: keep tracing but omit sensitive trace content:
# OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false

# Optional Browse and Flight Review configuration
FLIGHT_LOG_BROWSE_DB_PATH=outputs/browse.sqlite
FLIGHT_REVIEW_STORAGE_PATH=/path/to/flight_review_storage

# Explicit paths override the matching path derived from storage:
# FLIGHT_REVIEW_DB_PATH=/path/to/logs.sqlite
# FLIGHT_REVIEW_LOG_DIR=/path/to/log_files
# AIRFRAME_IMAGE_ROOT=/path/to/airframe/svg/assets
```

`.env` is ignored by Git. `scripts/start_web.sh` loads it automatically. The
direct analyzer CLI does not load `.env`; export or source the variables first.
If `OPENAI_MODEL` is unset, the current code defaults to `gpt-5.6`.

## Run the Web UI

```bash
./scripts/start_web.sh
```

Open <http://127.0.0.1:8000/>.

The main routes are:

- `/upload` — upload a ULog or select a server-local log, with optional mission,
  parameter XML, and PX4 source inputs. HTTP uploads are capped at 250 MiB.
- `/browse` — search, filter, sort, tag, download, and synchronize indexed logs.
- `/review?browse_id=<id>` — inspect the log, plots, tags, and analysis history,
  and ask an independent question about the ULog.

Additional server options are passed through the launcher:

```bash
./scripts/start_web.sh \
  --host 127.0.0.1 \
  --port 8000 \
  --flight-review-storage-path /path/to/flight_review_storage
```

The server also supports `--browse-db-path`, `--flight-review-db-path`,
`--flight-review-log-dir`, and `--airframe-image-root`.

## Run the Analyzer Directly

Load the API environment when using the CLI:

```bash
set -a
source .env
set +a
```

Then run the active v2 analyzer:

```bash
.venv/bin/python -m flight_log_agent.analysis.analyzer \
  --ulog /path/to/flight.ulg \
  --question "Why did the aircraft behave this way?" \
  --source ref/PX4-Autopilot \
  --output-dir outputs/manual_run
```

`--source` is optional when the initialized `ref/PX4-Autopilot` submodule is
available. Useful optional arguments include:

| Argument | Purpose |
| --- | --- |
| `--mission PATH` | Supply an optional mission file. |
| `--instructions PATH` | Append project-specific runtime instructions. |
| `--model MODEL` | Override `OPENAI_MODEL`. |
| `--max-turns N` | Set the analysis turn limit; default is 30. |
| `--max-total-requests N` | Apply an additional request cap. |
| `--no-web-fallback` | Remove hosted web search from the run. |
| `--web-context low\|medium\|high` | Select hosted web-search context size. |

Web search is available to the agent from the beginning of a default run. The
runtime prompt tells the agent to use it only when the local ULog and exact PX4
source leave a material external-context gap.

The turn, request, and web-search flags above apply to the direct CLI and Python
API. The current web UI uses the analyzer defaults: 30 turns and web search
available.

## Source and Shell Safety Model

Agent-generated commands run in a Bubblewrap environment with no network and
with these boundaries:

- The ULog and optional mission are mounted read-only under `/inputs/`.
- Persistent per-run scripts and intermediate files are writable under `work/`;
  `/tmp` is also writable but ephemeral.
- Plot artifacts are writable under `/plots/`.
- Python runs from the repository virtual environment.
- `rg`, `sed`, and `find` are limited to read-only workspace inspection.
- Pipes, redirection, and command chaining are rejected in top-level tool
  commands. Executing actions such as `find -exec` are also rejected.

PX4 source is not mounted into that workspace. Source inspection is handled
separately through commit-addressed Git operations:

```bash
git -C PX4-Autopilot grep -n -F "term" SNAPSHOT -- src/
git -C PX4-Autopilot show SNAPSHOT:src/module/file.cpp
git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/
```

`PX4-Autopilot` and `SNAPSHOT` are virtual values. The executor owns the
resolved repository and full commit SHA, rewrites the command, and permits only
`git grep`, `git show`, and `git ls-tree`. Nested submodule reads use the gitlink
commit recorded by the parent snapshot. These reads do not modify the PX4
branch, index, checkout, or dirty files, and no source copy is created.

The hosted OpenAI web-search tool is separate from the network-disabled shell
environment.

## Web Workflow and Analysis History

The frontend flow remains independent of the analyzer implementation:

1. Upload or select a ULog.
2. Preparse and index the log in the app-owned browse database.
3. Open the existing review workspace and lazy-load interactive plots.
4. Ask a question about the current log.
5. Poll the analysis run and display the structured report in the history
   timeline.

Every question creates an independent analysis run. Earlier questions and
answers are displayed for the user but are not automatically supplied as model
context to a later run.

## Storage Layout

Uploaded inputs and their durable history:

```text
uploads/<upload-id>/
├── <uploaded-name>.ulg
├── <optional-inputs>
└── analysis_history/
    └── <analysis-run-id>.json
```

Each web analysis has its own output directory:

```text
outputs/web_<analysis-run-id>/
├── report.json
├── plots/
└── work/
```

Other persistent application data:

```text
outputs/browse.sqlite
outputs/analysis_history/<stable-identity-digest>/<analysis-run-id>.json

.dev_logs/flight_agent_runs/<analysis-run-id>/
├── metadata.json
├── run_events.jsonl
└── usage.json
```

History for uploaded logs stays beside the uploaded ULog. History for indexed
external logs, including Flight Review logs, uses the stable fallback under
`outputs/analysis_history/`. Its directory digest is derived from the browse log
ID when available, otherwise from the resolved log path; it is not a hash of the
ULog contents.

`usage.json` records request and token usage reported by the SDK. It does not
calculate currency cost.

## Flight Review Database Synchronization

The Browse page can synchronize an existing Flight Review storage directory:

```text
<flight-review-storage>/
├── logs.sqlite
└── log_files/
```

Synchronization behavior:

- Opens the Flight Review SQLite database read-only.
- Writes normalized metadata only to this application's browse database.
- References existing ULog paths; it does not copy ULogs.
- Atomically adds new rows, updates changed rows, and leaves identical rows
  unchanged.
- Marks disappeared or missing source logs unavailable instead of deleting
  them, preserving tags and analysis history.
- Binds each browse database to one canonical Flight Review source to prevent
  identical IDs from different installations from overwriting one another.
- Rejects overlapping synchronization requests in the web process.

The storage path entered on the Browse page is a server-local, request-scoped
override of all configured Flight Review source paths and is not persisted.
Leaving it blank uses command-line or environment configuration. Within that
configuration, explicit database and log-directory paths take precedence over
the corresponding paths derived from `FLIGHT_REVIEW_STORAGE_PATH`.

See [Browse Page Design](docs/browse_page_design.md) for the complete behavior
and schema direction.

## Cost, Privacy, and Deployment Warnings

- Analysis calls a paid OpenAI model. Hosted web search can add usage.
- For direct CLI or Python API runs, use `--max-turns`,
  `--max-total-requests`, or `--no-web-fallback` when you need tighter usage
  limits. The web UI does not currently expose these per-run controls.
- ULog-derived content, prompts, and selected tool context are sent to the
  configured model provider as part of analysis.
- Developer audit logs can contain prompts, tool inputs, source excerpts, and
  analysis output. Treat `.dev_logs/` as sensitive local data.
- OpenAI Agents SDK tracing is not disabled by this branch. Set
  `OPENAI_AGENTS_DISABLE_TRACING=true`, or set
  `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=false` if you intentionally keep
  SDK tracing but do not want sensitive trace content included.
- Uploaded logs, generated scripts, reports, plots, and history under
  `uploads/` and `outputs/` can also contain sensitive flight-derived data.
- The web application has no authentication or authorization layer.
- Server-local upload paths, PX4 source paths, and Flight Review storage paths
  are intended for a trusted local deployment only.
- Do not expose the application with `--ngrok` unless an appropriate external
  authentication and authorization policy is in place.

If the ULog contains no firmware hash, or the local PX4 repository does not
contain that commit, the analyzer can still inspect the log but cannot make
source-confirmed claims. The report should reflect the resulting uncertainty.

## Tests

Run the complete suite through the repository virtual environment:

```bash
.venv/bin/python -m pytest
```

## Repository Map

| Path | Role |
| --- | --- |
| `flight_log_agent/analysis/analyzer.py` | Active single-agent analyzer and restricted shell executor. |
| `flight_log_agent/px4/source_snapshot.py` | Commit-addressed PX4 and submodule source access. |
| `flight_log_agent/ulog/` | ULog inventory, metrics, timeline, and interactive plot data. |
| `flight_log_agent/web/server.py` | Upload, browse, review, history, and analysis-run HTTP flow. |
| `flight_log_agent/web/browse_index.py` | App-owned browse database and Flight Review synchronization. |
| `flight_log_agent/models.py` | Stable structured `FlightLogReport` schema. |
| `web/` | Upload, browse, and review frontend. |
| `runner.py`, `flight_log_agent/runner_core.py` | Retained legacy runner path; not the active v2 web analyzer. |
| `tests/` | Unit and regression coverage. |

Additional reference documentation:

- [Browse Page Design](docs/browse_page_design.md)
- [PX4 Flight Review Plot Generation](docs/flight_review_plot_generation.md)
- [QGroundControl Airframe Asset Attribution](web/airframes/README.md)
