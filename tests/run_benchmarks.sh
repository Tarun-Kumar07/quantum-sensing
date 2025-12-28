#!/bin/bash
for i in {1..15}; do
  echo "Running test with $i threads..."
  pytest perf_test.py -s --junitxml="../data/perf_results_test_${i}_threads.xml" --num-threads="$i"
done
