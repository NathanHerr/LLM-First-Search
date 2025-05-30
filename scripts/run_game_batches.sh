#!/bin/bash

# Script to run game methods in separate tmux sessions for each batch
# Usage: ./run_game_batches.sh <game_type> <method> [options]

set -e

# Required parameters - to be specified by the user
GAME_TYPE=""
METHOD=""
DATA_DIR=""
OUTPUT_DIR=""  # New output directory parameter
CONDA_ENV=""
TOTAL_BATCHES=1  # Default number of batches to run
SESSION_PREFIX="game"  # Default session prefix
MODEL_NAME=""

# Function to display usage information
usage() {
  echo "Usage: $0 <game_type> <method> [options]"
  echo "  <game_type>            One of: countdown, sudoku"
  echo "  <method>               Search method (tot_bfs, bestfs, mcts, first_search)"
  echo "  Options:"
  echo "    --data-dir DIR       Data directory (REQUIRED)"
  echo "    --output-dir DIR     Output directory for results (REQUIRED)"
  echo "    --num-batches N      Number of batches to run (default: 2)"
  echo "    --session-prefix P   Tmux session prefix (default: game)"
  echo "    --conda-env ENV      Anaconda environment name (REQUIRED)"
  echo ""
  echo "  Game-specific options:"
  echo "    Countdown options:"
  echo "      --split SPLIT       Data split to use"
  echo "      --countdown-difficulty N  Game difficulty level"
  echo ""
  echo "    Sudoku options:"
  echo "      --sudoku-size N     Size of the Sudoku grid"
  echo "      --sudoku-difficulty DIFF  Difficulty level" 
  echo "      --sudoku-width N    Width of the Sudoku box"
  echo "      --sudoku-height N   Height of the Sudoku box"
  echo "      --num-puzzles N     Number of puzzles to solve"
  echo ""
  echo "  Model options:"
  echo "    --model-name NAME    Model name"
  echo "    --model-type TYPE    Model type: openai or nvidia"
  echo "    --is-azure N         Use Azure OpenAI endpoint (0=no, 1=yes)"
  echo "    --max-token-usage N  Maximum total token usage to allow before stopping"
  echo "    --timeout N          Timeout in seconds"
  echo "    --num-its N          Number of iterations"
  echo "    --batch-size N       Batch size"
  echo "    --reasoning N        Enable reasoning mode (0=disabled, 1=enabled)"
  echo ""
  echo "    --help               Display this help message"
  exit 1
}

# Function to get the Python script based on game type and method
get_script_path() {
  local game_type=$1
  local method=$2
  
  if [ "$game_type" = "countdown" ]; then
    case "$method" in
      "tot_bfs")
        echo "src.llm_tot_bfs"
        ;;
      "mcts")
        echo "src.llm_mcts"
        ;;
      "bestfs")
        echo "src.llm_bestfs"
        ;;
      "lfs")
        echo "src.llm_first_search"
        ;;
      *)
        echo "Unknown method for countdown: $method" >&2
        usage
        ;;
    esac
  elif [ "$game_type" = "sudoku" ]; then
    case "$method" in
      "tot_bfs")
        echo "src.llm_tot_bfs"
        ;;
      "mcts")
        echo "src.llm_mcts"
        ;;
      "bestfs")
        echo "src.llm_bestfs"
        ;;
      "lfs")
        echo "src.llm_first_search"
        ;;
      *)
        echo "Unknown method for sudoku: $method" >&2
        usage
        ;;
    esac
  else
    echo "Unknown game type: $game_type" >&2
    usage
  fi
}

# Parse command line arguments
if [ $# -lt 2 ]; then
  usage
fi

GAME_TYPE=$1
METHOD=$2
shift 2

# Additional command parameters to be passed to the Python script
EXTRA_PARAMS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      EXTRA_PARAMS="$EXTRA_PARAMS --output_dir $2"
      shift 2
      ;;
    --split)
      EXTRA_PARAMS="$EXTRA_PARAMS --split $2"
      shift 2
      ;;
    --countdown-difficulty)
      EXTRA_PARAMS="$EXTRA_PARAMS --countdown_difficulty $2"
      shift 2
      ;;
    --sudoku-size)
      EXTRA_PARAMS="$EXTRA_PARAMS --sudoku_size $2"
      shift 2
      ;;
    --sudoku-difficulty)
      EXTRA_PARAMS="$EXTRA_PARAMS --sudoku_difficulty $2"
      shift 2
      ;;
    --sudoku-width)
      EXTRA_PARAMS="$EXTRA_PARAMS --sudoku_width $2"
      shift 2
      ;;
    --sudoku-height)
      EXTRA_PARAMS="$EXTRA_PARAMS --sudoku_height $2"
      shift 2
      ;;
    --num-puzzles)
      EXTRA_PARAMS="$EXTRA_PARAMS --num_puzzles $2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"  # Store model name separately for session naming
      EXTRA_PARAMS="$EXTRA_PARAMS --model_name $2"
      shift 2
      ;;
    --model-type)
      EXTRA_PARAMS="$EXTRA_PARAMS --model_type $2"
      shift 2
      ;;
    --is-azure)
      EXTRA_PARAMS="$EXTRA_PARAMS --is_azure $2"
      shift 2
      ;;
    --batch-size)
      EXTRA_PARAMS="$EXTRA_PARAMS --batch_size $2"
      shift 2
      ;;
    --num-batches)
      TOTAL_BATCHES="$2"
      shift 2
      ;;
    --max-token-usage)
      EXTRA_PARAMS="$EXTRA_PARAMS --max_token_usage $2"
      shift 2
      ;;
    --timeout)
      EXTRA_PARAMS="$EXTRA_PARAMS --timeout $2"
      shift 2
      ;;
    --num-its)
      EXTRA_PARAMS="$EXTRA_PARAMS --num_its $2"
      shift 2
      ;;
    --reasoning)
      EXTRA_PARAMS="$EXTRA_PARAMS --reasoning $2"
      shift 2
      ;;
    --session-prefix)
      SESSION_PREFIX="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

# Validate game type and method
SCRIPT_PATH=$(get_script_path "$GAME_TYPE" "$METHOD")
if [ -z "$SCRIPT_PATH" ]; then
  exit 1
fi

# Check if required parameters are specified
if [ -z "$DATA_DIR" ]; then
  echo "Error: Data directory (--data-dir) must be specified."
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: Output directory (--output-dir) must be specified."
  exit 1
fi

# Check if Conda environment was specified
if [ -z "$CONDA_ENV" ]; then
  read -p "Anaconda environment name (required): " CONDA_ENV
  if [ -z "$CONDA_ENV" ]; then
    echo "Error: Anaconda environment name is required."
    exit 1
  fi
fi

# Print configuration values and ask for confirmation
echo "=== Game Batch Runner Configuration ==="
echo "Game Type:           $GAME_TYPE"
echo "Method:              $METHOD"
echo "Script Path:         $SCRIPT_PATH"
echo "Data Directory:      $DATA_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Number of Batches:   $TOTAL_BATCHES"
echo "Session Prefix:      $SESSION_PREFIX"
echo "Model Name:          $MODEL_NAME"
echo "Conda Environment:   $CONDA_ENV"
echo "Extra Parameters:    $EXTRA_PARAMS"
echo "============================================="

# Ask for confirmation
read -p "Are these settings correct? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
  echo "Aborting..."
  exit 1
fi

echo "Proceeding with batch creation..."
echo "================================="

# Create and run tmux sessions for each batch
for ((batch_num=0; batch_num<TOTAL_BATCHES; batch_num++)); do
  # Create a unique session name that includes the model name
  SESSION_NAME="${SESSION_PREFIX}_${GAME_TYPE}_${METHOD}_${MODEL_NAME}_batch_${batch_num}"
  
  # Ensure session name is valid for tmux (replace forward slashes with underscores)
  SESSION_NAME=$(echo "$SESSION_NAME" | tr '/' '_')
  
  # Check if the session already exists
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Skipping."
    continue
  fi
  
  # Construct the command to run
  CMD="conda activate $CONDA_ENV && python -m \"$SCRIPT_PATH\" --game_type $GAME_TYPE --search_method $METHOD --data_dir \"$DATA_DIR\" --batch_num $batch_num --output_dir \"$OUTPUT_DIR\" $EXTRA_PARAMS"
  
  # Create a new tmux session detached
  tmux new-session -d -s "$SESSION_NAME"
  
  # Send the command to the tmux session
  tmux send-keys -t "$SESSION_NAME" "cd \"$(pwd)\" && $CMD" C-m
  
  echo "Started batch $batch_num in tmux session: $SESSION_NAME"
done

echo ""
echo "All batches started. To attach to a session, use:"
echo "  tmux attach-session -t ${SESSION_PREFIX}_${GAME_TYPE}_${METHOD}_${MODEL_NAME}_batch_N  (where N is the batch number)"
echo ""
echo "To list all sessions:"
echo "  tmux list-sessions"
echo ""
echo "To kill a specific session:"
echo "  tmux kill-session -t ${SESSION_PREFIX}_${GAME_TYPE}_${METHOD}_${MODEL_NAME}_batch_N"
echo ""
echo "To kill all sessions for this method:"
echo "  tmux list-sessions | grep \"${SESSION_PREFIX}_${GAME_TYPE}_${METHOD}_${MODEL_NAME}\" | cut -d: -f1 | xargs -n1 tmux kill-session -t" 