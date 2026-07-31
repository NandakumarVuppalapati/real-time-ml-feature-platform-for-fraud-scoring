#!/usr/bin/env bash
# Submits flink_feature_job.py to the cluster, cancelling any existing run
# of the same job first so a container restart (e.g. after a code fix)
# doesn't leave two competing instances consuming the same Kafka consumer
# group -- which happened during development and silently starved one
# instance of all its input partitions.
set -u

JM="flink-jobmanager:8081"
JOB_NAME="fraud-card-feature-job"

echo "Waiting for jobmanager at $JM..."
until flink list -m "$JM" > /tmp/joblist.txt 2>&1; do
  echo "jobmanager not ready yet, retrying in 5s..."
  sleep 5
done

for JOB_ID in $(grep "$JOB_NAME" /tmp/joblist.txt | grep -oE '[0-9a-f]{32}'); do
  echo "Cancelling stale job $JOB_ID before resubmitting"
  flink cancel -m "$JM" "$JOB_ID" || true
done

until flink run -d -m "$JM" -py /opt/flink/usrlib/flink_feature_job.py; do
  echo "submission failed, retrying in 5s..."
  sleep 5
done

echo "Job submitted successfully."
