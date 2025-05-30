# 🧠 LLM-First Search (LFS): Self-Guided Search with Language Models

This repository accompanies our paper introducing **LLM-First Search (LFS)**, a novel self-guided search method that empowers Large Language Models (LLMs) to autonomously navigate and control the search process during problem-solving.

Unlike traditional search strategies that rely on fixed heuristics or hand-tuned exploration parameters (e.g., MCTS, BFS, BestFS), LFS puts the LLM in charge: the model itself decides whether to continue down a path or explore alternatives, guided by its own internal reasoning.

LFS enables **more flexible, efficient, and adaptive reasoning** without the need for manual hyperparameter tuning or domain-specific search strategies. It is especially effective for tasks that vary in difficulty or require dynamic compute allocation.

We evaluate LFS on two reasoning-intensive domains, **Countdown** and **Sudoku**, and compare it against three popular search-based approaches:

* Tree-of-Thoughts' **Breadth-First Search (ToT-BFS)**
* **Best-First Search (BestFS)**
* **Monte Carlo Tree Search (MCTS)**

## 🔍 Key Results

* ✅ Outperforms traditional search strategies on **harder tasks** without tuning.
* ⚡ Achieves **greater computational efficiency**, especially with stronger LLMs.
* 📈 **Scales better** with model size and compute budget, benefiting from its LLM-centric design.

## 🚀 Quick Start

```bash
# —— Option 1: Using Conda —— #
# Create & activate a Conda environment
conda create -n llmfirst python pip
conda activate llmfirst

# Install dependencies
pip install -r requirements.txt


# —— Option 2: Using pip + venv —— #
# Create & activate a virtual environment
python3 -m venv llmfirst
source llmfirst/bin/activate      # On Windows: llmfirst\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root directory with your OpenAI API key:

```bash
# Create .env file
touch .env

# Add your OpenAI API key to the .env file
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
```

**Note:** Replace `your_openai_api_key_here` with your actual OpenAI API key. Make sure to keep this file secure and never commit it to version control.

## 📊 Data Generation

### Countdown Data Generation

Generate Countdown puzzles with the following command:

```bash
python -m src.generators.countdown_generate \
    --seed 42 \
    --data_dir ./data/countdown/ \
    --list_length 3 \
    --max_target 100 \
    --num_samples 1000 \
    --val_ratio 0.2 \
    --check_duplicates
```

The script will generate:

* Training and validation sets split according to `val_ratio`
* Each sample includes starting numbers, target number, solution, and complexity score
* Files are saved as JSON in the specified data directory

Example of a single Countdown puzzle data structure:

```json
{
  "nums": [60, 96, 84, 22, 5, 26, 30],
  "target": 75,
  "solution": [
    "60+96=156",
    "84+22=106",
    "30-5=25",
    "156/26=6",
    "106-25=81",
    "81-6=75"
  ],
  "complexity": 0.6,
  "optimal_path": "...trace...",
  "start_size": 7
}
```

### Sudoku Data Generation

Generate Sudoku puzzles with the following command:

```bash
python -m src.generators.sudoku_generator \
    --difficulty expert \
    --width 3 \
    --height 3 \
    --number 100 \
    --show-solutions 0 \
    --output ./data/sudoku/ \
    --save 1
```

The script will generate:

* Puzzles with unique solutions
* Different difficulty levels affect the number of empty cells
* Files are saved as pickle files in the specified output directory
* Each puzzle includes the board state and solution

Example of a Sudoku puzzle and its solution:

```
# Example 6x6 Sudoku puzzle (width=2, height=3)
Puzzle:
[
    [None, 3, None, 1, None, None],
    [1, None, None, None, None, None],
    [None, None, 3, None, None, 2],
    [None, None, None, None, None, None],
    [None, 2, None, 6, None, None],
    [6, 1, None, 4, None, 5]
]

Solution:
[
    [2, 3, 6, 1, 5, 4],
    [1, 5, 4, 2, 6, 3],
    [4, 6, 3, 5, 1, 2],
    [5, 4, 1, 3, 2, 6],
    [3, 2, 5, 6, 4, 1],
    [6, 1, 2, 4, 3, 5]
]
```

## 🌎 Game Architecture

The codebase is modular and designed for easy extension to new games. It currently supports two games with dedicated modules:

### Game Modules

* **`countdown_game`**: Contains Countdown-specific node and agent implementations. Utilities are in **`utils.countdown_utils`**.
* **`sudoku_game`**: Contains Sudoku-specific node and agent implementations. Utilities are in **`utils.sudoku_utils`**.

### Node Types

Each game defines a custom node type that inherits from a common **`GameNode`** base class, ensuring all node types implement a consistent interface required by the search algorithms.

### Game-Specific Agents

Each game includes an agent inheriting from **`BaseAgent`**, using **`BaseInstructions`** for prompt handling. These agents manage:

* API calls to LLMs
* Prompt formatting and parsing
* Game state transitions
* Move generation and validation

This modular structure allows new games to be added with minimal changes to the search logic.

### Search Algorithms

Search strategies are implemented in separate files with game-specific run logic:

* **`llm_first_search.py`**: LLM-First Search
* **`llm_mcts.py`**: Monte Carlo Tree Search
* **`llm_bestfs.py`**: Best-First Search
* **`llm_tot_bfs.py`**: Tree-of-Thoughts BFS

Each script contains a `run` function tailored to the game domain, maintaining a unified interface across algorithms.

## 🔮 Running Experiments

### Countdown Example

```bash
./scripts/run_game_batches.sh countdown mcts \
    --data-dir ./data/countdown \
    --output-dir ./data/countdown/results \
    --conda-env llmfirst \
    --model-type openai \
    --model-name o3-mini \
    --reasoning 1 \
    --split val \
    --countdown-difficulty 3 \
    --num-batches 1 \
    --max-token-usage 10000 \
    --timeout 300 \
    --num-its 1 \
    --batch-size 1 \
    --session-prefix game
```

### Sudoku Example

```bash
./scripts/run_game_batches.sh sudoku lfs \
    --data-dir ./data/sudoku \
    --output-dir ./data/sudoku/results \
    --conda-env llmfirst \
    --model-type openai \
    --model-name gpt-4o \
    --reasoning 0 \
    --sudoku-difficulty medium \
    --sudoku-size 4 \
    --sudoku-width 2 \
    --sudoku-height 2 \
    --num-batches 1 \
    --max-token-usage 10000 \
    --timeout 300 \
    --num-its 1 \
    --batch-size 1 \
    --session-prefix game
```

## 📊 Analysis

After running experiments, you can analyze the results using the included analysis script. The script automatically detects available methods and tasks from your result directories and generates comprehensive comparisons.

### Basic Analysis

```bash
# Auto-detect all methods and tasks
python src/simple_analysis.py \
    --data_dir ./data/countdown/results \
    --output_dir ./analysis/countdown

# Analyze specific model
python src/simple_analysis.py \
    --data_dir ./data/sudoku/results \
    --output_dir ./analysis/sudoku \
    --model gpt-4o

# Include only specific methods
python src/simple_analysis.py \
    --data_dir ./data/countdown/results \
    --output_dir ./analysis/countdown \
    --methods "mcts,bestfs,lfs"
```

### Generated Outputs

The analysis script automatically generates:

#### 📈 Visualizations
* **`win_rates.png`**: Bar chart comparing win rates across methods and tasks
* **`token_usage.png`**: Bar chart comparing average token usage across methods
* **`efficiency.png`**: Scatter plot showing win rate vs token usage (efficiency analysis)

#### 📋 Summary Tables
* **`summary.xlsx`**: Detailed Excel spreadsheet with all metrics
* **`summary.csv`**: CSV version of the summary data

#### 🖥️ Console Output
The script prints a detailed summary to the console:

```
📊 SUMMARY:
================================================================================

Countdown (diff=3):
----------------------------------------
  mcts        : 85.0% win rate, 1250 tokens
  bestfs      : 78.0% win rate, 950 tokens
  lfs         : 92.0% win rate, 1100 tokens
  tot_bfs     : 70.0% win rate, 1400 tokens

Sudoku (4x4, medium):
----------------------------------------
  mcts        : 60.0% win rate, 2100 tokens
  bestfs      : 55.0% win rate, 1800 tokens
  lfs         : 75.0% win rate, 1900 tokens
```

### Key Metrics

The analysis automatically computes and compares:

* **Win Rate**: Percentage of games solved successfully
* **Token Usage**: Average number of tokens consumed per game
* **Efficiency**: Win rate relative to computational cost
* **Game Coverage**: Number of games attempted and solved

### Analysis Tips

* Run analysis on specific subsets using the `--methods` parameter to focus comparisons
* Compare the same model across different tasks to understand method strengths
* Use the efficiency plots to identify methods that achieve good win rates with lower token costs
* The Excel output provides detailed breakdowns for further statistical analysis
