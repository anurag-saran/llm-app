# AIFirst-LLM-benchmarking

## Repository structure

**detectors** - Detectors APIs

**orchestrator** - Benchmark runner

## How to run the benchmark locally

### Prerequisites
The following tools have to be installed:
* podman
* podman-compose

In `orchestrator` directory, copy `.env.example` file to `.env` 
and adjust the values if needed (like setting API keys for models).

### Building images
The following command builds images for the Orchestrator and Detectors.

```bash
./build.sh
```

### Running benchmark environment
This includes the Detectors, MongoDB, and MongoDB UI.
The command below runs the environment and shows the logs  

```bash
./podman-up.sh

```

For the first run it will also build the Detectors images

Press Ctrl+C to terminate the logs. If you need the logs again, run the following command:
```bash
podman-compose logs -f
```

### Running benchmark
Before running the benchmark, the prompts DB should be populated with prompts.
```bash
./run.sh --init-db
```

The following command runs the benchmark.

```bash
# Run benchmark
./run.sh --llm-type openai --model gpt-3.5-turbo
```

`llm-type` and `model` parameters are required.

### Shut down the environment
You can reuse benchmarking environment to run multiple benchmarks.

If you need to shut it down, terminate the logs and run:

```bash
podman-compose down
```
# llm-app
