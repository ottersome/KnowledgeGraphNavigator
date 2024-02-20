#!/bin/bash

TTL_FILES_PATH=$1
TDB_LOC=$2

ttl_files=$(find $TTL_FILES_PATH -name '*.ttl')
echo "Discovered files : ${ttl_files}"

for ttlFile in $ttl_files; do
  echo "Translating ${ttlFile}"
  graphName="http://huginns.io/graph/$(basename "$ttlFile" .ttl)"
  tdb2.tdbloader --loc=$TDB_LOC --graph=$graphName "$ttlFile"
done
