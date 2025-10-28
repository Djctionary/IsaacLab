#!/bin/bash

# Batch performance test script
# For testing different gait configurations of Unitree Go2 models

# Set base paths
BASE_LOG_PATH="logs/rsl_rl/unitree_go2_rough"
SCRIPT_PATH="scripts/reinforcement_learning/rsl_rl/play_performance_test.py"

# Define experiment configuration array
# Format: "run_id:gait_type:gait_freq:checkpoint"
declare -a experiments=(
    "2025-10-27_17-53-28:no_gait:0:model_2200.pt"
    # "2025-10-27_22-27-34:no_gait:0:model_2000.pt"
    # "2025-10-27_02-19-59:gait_3:3:model_2450.pt"
    # "2025-10-27_12-22-56:gait_3:3:model_2500.pt"
    # "2025-10-27_22-27-00:gait_3:3:model_1800.pt"
    # "2025-10-27_12-24-15:gait_5:5:model_1800.pt"
    # "2025-10-26_23-44-07:gait_5:5:model_1800.pt"
    # "2025-10-26_22-37-25:gait_5:5:model_1700.pt"
    # "2025-10-27_15-56-36:gait_7:7:model_1800.pt"
    # "2025-10-27_02-22-03:gait_7:7:model_550.pt"
)

# Check if all checkpoint files exist
echo "Checking checkpoint files..."
missing_files=0

for exp in "${experiments[@]}"; do
    IFS=':' read -r run_id gait_type gait_freq checkpoint <<< "$exp"
    checkpoint_path="${BASE_LOG_PATH}/${run_id}/${checkpoint}"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "Error: checkpoint file not found: $checkpoint_path"
        missing_files=1
    else
        echo "✓ Found: $checkpoint_path"
    fi
done

# Exit script if any files are missing
if [ $missing_files -eq 1 ]; then
    echo "Missing checkpoint files detected, script terminated."
    exit 1
fi

echo "All checkpoint files verified, starting batch testing..."
echo "================================================"

# Execute batch testing
for exp in "${experiments[@]}"; do
    IFS=':' read -r run_id gait_type gait_freq checkpoint <<< "$exp"
        # "2025-10-27_22-27-34:no_gait:0:model_2000.pt"
    # "2025-10-27_02-19-59:gait_3:3:model_2450.pt"
    # "2025-10-27_12-22-56:gait_3:3:model_2500.pt"
    # "2025-10-27_22-27-00:gait_3:3:model_1800.pt"
    # "2025-10-27_12-24-15:gait_5:5:model_1800.pt"
    # "2025-10-26_23-44-07:gait_5:5:model_1800.pt"
    # "2025-10-26_22-37-25:gait_5:5:model_1700.pt"
    # "2025-10-27_15-56-36:gait_7:7:model_1800.pt"
    # "2025-10-27_02-22-03:gait_7:7:model_550.pt"
    echo ""
    echo "Starting test: $run_id ($gait_type, freq=$gait_freq)"
    echo "Checkpoint: $checkpoint"
    echo "----------------------------------------"
    
    # Select task and parameters based on gait type
    if [ "$gait_type" = "no_gait" ]; then
        task="Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
        cmd="./isaaclab.sh -p $SCRIPT_PATH --task=$task --checkpoint=${BASE_LOG_PATH}/${run_id}/${checkpoint} --iterations=1 --headless"
    else
        task="Isaac-Velocity-Rough-Phase-Unitree-Go2-Play-v0"
        cmd="./isaaclab.sh -p $SCRIPT_PATH --task=$task --checkpoint=${BASE_LOG_PATH}/${run_id}/${checkpoint} --gait_freq=$gait_freq --iterations=1 --headless"
    fi
    
    echo "Executing command: $cmd"
    
    # Execute command
    eval $cmd
    
    # Check command execution result
    if [ $? -eq 0 ]; then
        echo "✓ $run_id test completed"
    else
        echo "✗ $run_id test failed"
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "================================================"
echo "Batch testing completed!"
