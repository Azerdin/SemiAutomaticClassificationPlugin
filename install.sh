#!/bin/bash
if [ -z "$1" ]; then
    echo "Missing parameter with user name"
    exit 1
fi
path="/Users/$1/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/SemiAutomaticClassificationPlugin"

rm -rf "$path"
cp -R "$PWD" "$path"
exit 0;