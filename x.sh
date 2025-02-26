#!/bin/bash

# Step 1: Change directory to cs160
echo "Changing directory to cs160..."
cd cs160 || { echo "Failed to cd into cs160"; exit 1; }

# Step 2: Pull latest changes from git
echo "Pulling latest changes from git..."
git pull || { echo "Git pull failed"; exit 1; }

# Step 3: Wait for 3 seconds
echo "Waiting for 3 seconds..."
sleep 2

# Step 4: Change directory back to parent
echo "Changing directory back to parent..."
cd .. || { echo "Failed to cd into parent directory"; exit 1; }


echo "Copying new-forward-kernel to cse160-WI25/PA6..."
cp cs160/new-forward-kernel.cl cse160-WI25/PA6/src/layer/custom/ || { echo "Failed to copy new-forward-kernel.cl"; exit 1; }


echo "Copying new-forward.cc to cse160-WI25/PA6..."
cp cs160/new-forward.cc cse160-WI25/PA6/src/layer/custom/ || { echo "Failed to copy new-forward.cc"; exit 1; }

echo "All steps completed successfully!"
