#!/bin/bash

export REGISTRY="registry.hub.docker.com"
export PROJECT="emeryaifirst"
export IMAGE_PREFIX=$REGISTRY/$PROJECT
export IMAGE_PREFIX_LOCAL=aifirst-llm-benchmarking_

export IMAGE_LLM_BENCHMARK=llm-benchmark
export IMAGE_DETECTOR_TOXICITY=detector-toxicity
export IMAGE_DETECTOR_RELEVANCE=detector-relevance
export IMAGE_DETECTOR_HALLUCINATIONS=detector-hallucinations
export IMAGE_DETECTOR_PRIVACY=detector-privacy

echo  "Pushing images into: $IMAGE_PREFIX"

podman tag $IMAGE_LLM_BENCHMARK $IMAGE_PREFIX/$IMAGE_LLM_BENCHMARK
podman tag $IMAGE_PREFIX_LOCAL$IMAGE_DETECTOR_TOXICITY $IMAGE_PREFIX/$IMAGE_DETECTOR_TOXICITY
podman tag $IMAGE_PREFIX_LOCAL$IMAGE_DETECTOR_RELEVANCE $IMAGE_PREFIX/$IMAGE_DETECTOR_RELEVANCE
podman tag $IMAGE_PREFIX_LOCAL$IMAGE_DETECTOR_HALLUCINATIONS $IMAGE_PREFIX/$IMAGE_DETECTOR_HALLUCINATIONS
podman tag $IMAGE_PREFIX_LOCAL$IMAGE_DETECTOR_PRIVACY $IMAGE_PREFIX/$IMAGE_DETECTOR_PRIVACY

podman push $PROJECT/$IMAGE_LLM_BENCHMARK
podman push $PROJECT/$IMAGE_DETECTOR_TOXICITY
podman push $PROJECT/$IMAGE_DETECTOR_RELEVANCE
podman push $PROJECT/$IMAGE_DETECTOR_HALLUCINATIONS
podman push $PROJECT/$IMAGE_DETECTOR_PRIVACY