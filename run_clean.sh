#!/bin/bash

# Remove the __pycache__ folder if it exists
if [ -d "__pycache__" ]; then
    echo "Removing __pycache__ folder..."
    rm -rf __pycache__
else
    echo "__pycache__ folder not found."
fi

# Remove feat.bin if it exists
if [ -f "feat.bin" ]; then
    echo "Removing feat.bin..."
    rm feat.bin
else
    echo "feat.bin not found."
fi

# Remove the super_vertex folder if it exists
if [ -d "super_vertex" ]; then
    echo "Removing super_vertex folder..."
    rm -rf super_vertex
else
    echo "super_vertex folder not found."
fi

echo "Clean up completed."
