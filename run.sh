#!/bin/bash


podman run --network=aifirst-llm-benchmarking_default -p 8080:80 \
    -e MONGO_URI=mongodb://mongodb:27017/benchmark \
    --env-file=orchestrator/.env \
    -v ./results:/app/results \
    --rm --name llm-benchmark llm-benchmark:latest "$@"
