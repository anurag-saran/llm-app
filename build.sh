#!/bin/bash

podman-compose build
podman build --tag llm-benchmark:latest -f ./orchestrator/Dockerfile
