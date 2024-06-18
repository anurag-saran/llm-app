#!/bin/bash

podman-compose --env-file=orchestrator/.env up -d
podman-compose logs -f
