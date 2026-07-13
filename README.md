# Benchmark Using Multi-LLMs for Tokenomics

This repository is designed to evaluate various Large Language Model (LLM) combinations and cascading strategies across multiple code-generation use cases.

The project explores how to balance benchmark performance (e.g., code correctness) with API cost using Google Cloud Vertex AI and Anthropic APIs.

---

## Repository Structure

```
.
├── .gitignore
├── MODELS.md               # Details of supported models and pricing configurations
├── README.md               # This file
├── requirements.txt        # Python package dependencies
├── src/                    # Shared Python library
│   ├── __init__.py
│   ├── config.py           # Model settings, prices, and prompt templates
│   ├── client.py           # Vertex AI SDK client wrappers and backoff retry logic
│   ├── evaluator.py        # Code extraction and sandboxed unit test runner
│   └── architectures.py    # Standard evaluation architectures (Single, Read/Write, etc.)
├── bigCodeBench-hard/      # Use Case 1: BigCodeBench-Hard code generation tasks
│   ├── data/               # HF downloaded dataset
│   ├── results/            # Run metric logs
│   ├── bench_runner.py     # Benchmark runner script for BigCodeBench-Hard
│   └── build_html_dashboard.py
└── webdev/                 # Use Case 2: Web/Networking development tasks
    ├── data/               # Local WebDev dataset
    ├── results/            # Run metric logs
    └── bench_runner.py     # Benchmark runner script for Web-Dev
```

---

## Getting Started

### 1. Setup Virtual Environment

Create a clean python virtual environment (`tokenomics-bench-env`) and install dependencies:

```bash
# Create virtual environment
python3 -m venv tokenomics-bench-env

# Activate virtual environment
source tokenomics-bench-env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Google Cloud Credentials

Ensure your Application Default Credentials (ADC) are configured for Vertex AI:

```bash
gcloud auth application-default login
```

Set the GCP project and location where your Vertex AI models are deployed:

```bash
export GCP_PROJECT="your-gcp-project-id"
export GCP_LOCATION="us-central1" # or other region supporting Gemini 3.5
```

---

## Running Benchmarks

### Use Case 1: BigCodeBench-Hard

Run a single architecture evaluation (e.g., `hybrid`) on the first 10 tasks:
```bash
./tokenomics-bench-env/bin/python bigCodeBench-hard/bench_runner.py --arch hybrid --n 10
```

Compare all standard configurations:
```bash
./tokenomics-bench-env/bin/python bigCodeBench-hard/bench_runner.py --compare-all --n 10
```

#### Available Architectures
- `single`: Direct completion by one model (e.g., `gemini-3.5-flash`).
- `read-write`: Advisor-Executor split (Planner + Executor).
- `cascade`: Offload generation to a cheap model, repair with a premium thinking model.
- `hybrid`: Custom hybrid pipeline combining planning, triage, cheap repair, and thinking escalation.

---

### Use Case 2: Web Development

Run the WebDev benchmark on web-related libraries filtered from BigCodeBench-Hard:
```bash
./tokenomics-bench-env/bin/python webdev/bench_runner.py --arch hybrid --n 5
```

Compare all architectures including advanced routing configurations:
```bash
./tokenomics-bench-env/bin/python webdev/bench_runner.py --compare-all --n 5
```

#### Web-Specific Architectures
- `router`: Dynamic routing based on complexity prediction.
- `dual-advisor`: Separates algorithmic planning from API contract planning.
- `tdd`: Generates test cases first to guide the executor.
- `shield`: Multi-tier fallback validation.
- `peer-reviewer`: Self-correction loop via an independent auditor model.

---

## Detailed Model Configuration
For more information on the models evaluated, pricing rates, and detailed architecture pipelines, please refer to [MODELS.md](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/MODELS.md).

---

## Extending the Benchmark

You can easily add new models or define custom benchmark architectures.

### 1. Adding a New Model
1. Open [src/config.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/src/config.py).
2. Define your Model ID constant.
3. Add the pricing entry (USD per 1,000,000 tokens) under `PRICING` (for input, output, and optional cache read/write rates).

### 2. Defining a Custom Architecture
1. Open [src/architectures.py](file:///Users/lexha/Documents/work/codes/prj/17-tokenomics/benchmark-using-multi-LLMs/src/architectures.py).
2. Create a new function (e.g., `run_my_custom_flow(problem, ...)`).
3. Use `dispatch_model(model_id, prompt)` to orchestrate LLM calls, `extract_code(text)` to clean outputs, and `run_bigcodebench(problem, code)` to execute unit tests.
4. Return a dict matching the metrics schema:
   ```python
   {
       "passed": True/False,
       "as_run_usd": 0.00,
       "output_tokens": 120,
       "total_tokens": 500,
       "error": "..."
   }
   ```

### 3. Exposing the Architecture in Runners
1. Open the runner script (e.g., `bigCodeBench-hard/bench_runner.py` or `webdev/bench_runner.py`).
2. Add your new architecture choice to `parser.add_argument("--arch", choices=[...])`.
3. Map the choice inside the `run_benchmark` driver function to invoke your custom architecture function.

