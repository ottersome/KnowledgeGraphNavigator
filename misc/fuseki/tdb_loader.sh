#!/bin/bash

# Set by command line arg
TTL_PATH=$1
# Take filename from TTL_PATH and remove extension

# Location of your TDB database
TDB_LOCATION=$2

# Load each TTL file into the TDB database
for ttl_file in "$TTL_PATH"/*.ttl; do
    TTL_FILENAME=$(basename "$ttl_file" .ttl)
    echo "Loading $ttl_file into TDB database..."
    tdbloader --loc="$TDB_LOCATION/$TTL_FILENAME" "$ttl_file"
done

echo "All TTL files have been loaded into the TDB database."

