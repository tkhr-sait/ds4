#!/bin/bash

for PREFIX in nommap- residency-lru- baseline-
do
  ENV=""
  if [ "${PREFIX}" = "nommap-" ]; then
    ENV="DS4_METAL_ENABLE_STREAMING_NO_MMAP=1"
  elif [ "${PREFIX}" = "residency-lru-" ]; then
    #ENV="DS4_METAL_ENABLE_STREAMING_RESIDENCY_LRU=1 DS4_RESIDENCY_LRU_PROFILE=1"
    ENV="DS4_METAL_ENABLE_STREAMING_RESIDENCY_LRU=1"
  fi
  for TAG in q2 q4
  do
    for BUDGET_MIB in 8192 16384 32768 65536 81920
    do
      echo "-------------------------------------------"
      echo PREFIX=${PREFIX} BUDGET_MIB=${BUDGET_MIB} TAG=${TAG} "${ENV}"
      echo "-------------------------------------------"
      PREFIX=${PREFIX} BUDGET_MIB=${BUDGET_MIB} TAG=${TAG} DS4_SERVER_BIN=../../ds4-server bash ../scripts/q4-experiment.sh pread "${ENV}"
      sleep 10
    done
  done
done
