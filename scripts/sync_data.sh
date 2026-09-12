#!/bin/bash
# Push the demo shards to the training server. 19 GB, so --partial: rerun after
# a dropped connection and rsync resumes rather than starting the file over.
set -euo pipefail

REMOTE=lab:/home/bobcowher/data/farama-kitchen-bc/dataset/

ssh "${REMOTE%%:*}" mkdir -p "${REMOTE#*:}"
rsync -a --partial --info=progress2 "$(dirname "$0")/../dataset/" "$REMOTE"
