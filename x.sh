#!/bin/bash

# Step 1: Change directory to cs160
echo "Changing directory to cs160..."
cd cs160 || { echo "Failed to cd into cs160"; exit 1; }

# Step 2: Pull latest changes from git
echo "Pulling latest changes from git..."
git pull || { echo "Git pull failed"; exit 1; }

# Step 3: Wait for 3 seconds
echo "Waiting for 3 seconds..."
sleep 3

# Step 4: Change directory back to parent
echo "Changing directory back to parent..."
cd .. || { echo "Failed to cd into parent directory"; exit 1; }

# Step 5: Copy main.c to cse160-WI25/PA5
echo "Copying main.c to cse160-WI25/PA5..."
cp cs160/main.c cse160-WI25/PA5 || { echo "Failed to copy main.c"; exit 1; }

# Step 6: Copy kernel.cl to cse160-WI25/PA5
echo "Copying kernel.cl to cse160-WI25/PA5..."
cp cs160/kernel.cl cse160-WI25/PA5 || { echo "Failed to copy kernel.cl"; exit 1; }

echo "All steps completed successfully!"
