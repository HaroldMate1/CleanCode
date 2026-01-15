# Tecnun-IAthon ResearchAssistant

An AI-powered **data exploration and reporting tool** designed for the Tecnun-IAthon hackathon. Give it a CSV/Excel file, and it will automatically:

- ✅ Validate and clean the data
- ✅ Iteratively explore it using AI "agents"
- ✅ Generate multiple visualizations and statistical tests
- ✅ Produce a comprehensive Markdown report (optionally converted to PDF, DOCX, or PPTX)
- ✅ Track and summarize OpenAI token usage and cost

This is a **ReAct-style agentic framework** that combines planning, code generation, and safe execution for intelligent data analysis.

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Git Clone](#git-clone)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Components](#components-code-overview)
- [Troubleshooting](#troubleshooting)
- [Hackathon Notes](#hackathon-notes-tecnun-iathon)
- [License](#license)

---

## 🚀 Quick Start

Get started in 3 minutes:

```bash
# Clone the repository
git clone https://github.com/arubio2/CleanCode.git
cd CleanCode

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."  # Windows: set OPENAI_API_KEY=sk-...

# Run analysis
python AgentResearchAssitant.py --input data.csv --output-dir results/ --format pptx
```

---

## 📦 Git Clone

### Clone the Repository

```bash
git clone https://github.com/arubio2/CleanCode.git
cd CleanCode
```

### Clone with SSH (if configured)

```bash
git clone git@github.com:arubio2/CleanCode.git
cd CleanCode
```

---

## ✨ Features

### Core Capabilities

- **Command-line tool**: Simple CLI interface for running full analyses
- **Data loading & cleaning**:
  - Reads CSV/Excel into a Pandas DataFrame
  - Drops low-information and constant columns
  - Identifies numeric vs. categorical features

- **Data Quality Validation**:
  - Checks for synthetic patterns (too-perfect uniform distributions)
  - Identifies outliers and domain constraint violations
  - Detects temporal logic issues
  - Prints human-readable validation report
  - Optionally auto-fixes some issues (negative times, invalid ages, etc.)

- **Safe execution sandbox** (SafeRunner):
  - Executes AI-generated Python code safely
  - Pre-loads `df`, `num_features`, `cat_features`
  - Access to `numpy`, `pandas`, `scipy.stats`, `matplotlib`, `seaborn`, `plotly`
  - Dedicated `figures/` directory for all outputs
  - Blocks file-reading functions (`pd.read_csv`, etc.)
  - Automatically redirects `plt.savefig()` calls to output directory
  - Ensures at least one `.png` is created or raises error

- **AI "Agents"**:
  - **DecisionMaker**: Plans analysis steps and synthesizes final report
  - **CodexGenerator**: Generates robust Python code for analysis and plotting
  - **ReActAnalyzer**: Orchestrates iterative loop between agents and SafeRunner

- **ReAct-style analysis loop**:
  - Plans next analysis step
  - Generates and runs code
  - Observes results and errors
  - Stops when analysis is complete

- **Report generation & conversion**:
  - Saves `report.md` in output directory
  - Optionally converts to PPTX, PDF, or DOCX via `pandoc`
  - Includes custom PowerPoint styling support

- **OpenAI cost tracking**:
  - Live pricing fetch (with caching and robust fallbacks)
  - Tracks tokens per model
  - Prints detailed cost summary at end
  - Saves `token_usage.log` file

---

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/arubio2/CleanCode.git
cd CleanCode
```

### Step 2: Create and Activate Virtual Environment

```bash
# Using venv (recommended)
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n cleancode python=3.10
conda activate cleancode
```

### Step 3: Install Dependencies

#### Minimum Requirements

```bash
pip install -r requirements.txt
```

If no `requirements.txt` exists, install these packages:

```bash
pip install pandas numpy scipy matplotlib seaborn plotly openai requests beautifulsoup4 python-pptx
```

#### Optional Dependencies

```bash
# For live pricing functionality (optional)
pip install requests beautifulsoup4

# For PowerPoint enhancements (optional)
pip install python-pptx

# For PDF/DOCX conversion (optional but recommended)
pip install pandoc
```

### Step 4: Install Pandoc (for document conversion)

Pandoc is required for PDF/DOCX/PPTX conversion:

- **macOS**: `brew install pandoc`
- **Windows**: Download from [https://pandoc.org/install.html](https://pandoc.org/install.html)
- **Linux**: `sudo apt-get install pandoc`

### Step 5: Verify Installation

```bash
python -c "import openai, pandas, numpy; print('✅ All packages installed!')"
```

---

## ⚙️ Configuration

### OpenAI API Key

The script expects an OpenAI API key via:

#### Option 1: Environment Variable (Recommended)

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...
```

#### Option 2: CLI Argument

```bash
python AgentResearchAssitant.py --api-key "sk-..." --input data.csv
```

---

## 📖 Usage

### Basic Usage

```bash
python AgentResearchAssitant.py \
  --input path/to/data.csv \
  --output-dir results/run1 \
  --format pptx \
  --max-steps 6 \
  --verbose
```

### Command-Line Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--input` | str | Path to CSV/Excel input file | **Required** |
| `--output-dir` | str | Output directory for results | `./results` |
| `--format` | str | Output format: `pptx`, `pdf`, `docx`, or `md` | `md` |
| `--max-steps` | int | Maximum analysis iterations | `6` |
| `--prompt` | str | Custom analysis prompt | (auto-generated) |
| `--api-key` | str | OpenAI API key | (from environment) |
| `--verbose` | flag | Enable verbose logging | `False` |

### Example Workflows

#### Workflow 1: Quick Analysis (Markdown only)
```bash
python AgentResearchAssitant.py --input sales_data.csv --output-dir analysis/
```

#### Workflow 2: Full Report (PowerPoint)
```bash
python AgentResearchAssitant.py \
  --input customer_data.csv \
  --output-dir analysis/customers \
  --format pptx \
  --max-steps 8
```

#### Workflow 3: PDF Report with Custom Prompt
```bash
python AgentResearchAssitant.py \
  --input experiment_results.csv \
  --output-dir analysis/experiments \
  --format pdf \
  --prompt "Focus on statistical significance and effect sizes" \
  --max-steps 5
```

### Output Structure

```
results/run1/
├── report.md                  # Markdown report
├── report.pdf                 # Converted PDF (if --format pdf)
├── report.docx                # Converted DOCX (if --format docx)
├── report.pptx                # Converted PowerPoint (if --format pptx)
├── figures/                   # Generated visualizations
│   ├── fig_001.png
│   ├── fig_002.png
│   └── ...
└── token_usage.log            # Cost & token summary
```

---

## 🏗️ Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────┐
│           CLI + Setup                               │
│  (args, output dir, UsageTracker, pricing)          │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│      Data Loading & Validation                      │
│  (load_and_clean, DataQualityValidator)             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│        Safe Execution Environment                   │
│  (SafeRunner with sandboxed exec)                   │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│     AI Agents (ReAct Loop)                          │
│  DecisionMaker → CodexGenerator → SafeRunner        │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│   Report Generation & Conversion                    │
│  (Markdown → PDF/DOCX/PPTX via pandoc)             │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│    Cost Summary & Logging                           │
│  (UsageTracker.print_summary & save_log)            │
└─────────────────────────────────────────────────────┘
```

### Pipeline Flow

**Step 1: CLI & Setup**
- Parse command-line arguments
- Create output directory structure
- Initialize `UsageTracker` (fetches live OpenAI pricing)

**Step 2: Data Loading & Validation**
- Load CSV/Excel into DataFrame via `load_and_clean()`
- Run `DataQualityValidator.validate()`:
  - Check synthetic patterns
  - Identify outliers
  - Verify domain constraints
  - Print validation report

**Step 3: SafeRunner Initialization**
- Create `SafeRunner` instance with cleaned DataFrame
- Prepare `figures/` subdirectory
- Configure safe execution environment

**Step 4: ReActAnalyzer Loop** (Orchestrator)
- For each step (up to `max_steps`):
  1. **Ask DecisionMaker**: What's the next analysis step?
  2. **Ask CodexGenerator**: Generate Python code for that step
  3. **Run with SafeRunner**: Execute code, capture figures & errors
  4. **Update State**: Log results and observations

**Step 5: Final Report & Conversion**
- Ask DecisionMaker to synthesize final Markdown report
- Save `report.md`
- If `--format` specified, convert via pandoc:
  - `pptx`: PowerPoint with optional custom styling
  - `pdf` / `docx`: Direct conversion
- Save `token_usage.log`

---

## 🧩 Components (Code Overview)

### UsageTracker

Tracks token usage and calculates costs:

```python
# Initialize
tracker = UsageTracker()

# Track a model call
tracker.track_tokens(model="gpt-4o", input_tokens=100, output_tokens=50)

# Get summary
tracker.print_summary()
tracker.save_log(filepath="token_usage.log")
```

**Features**:
- Fetches live pricing from OpenAI
- Caches pricing locally (`~/.cache/iathlon/openai_pricing.json`)
- Falls back to hardcoded table if fetch fails
- Supports text and image models

### SafeRunner

Executes AI-generated code safely:

```python
runner = SafeRunner(df, num_features, cat_features, figures_dir)
result = runner.run(code_string)
# Raises error if no .png files generated
```

**Security Features**:
- Blocks file-reading functions (`pd.read_*`)
- Injects safe global environment
- Rewrites `plt.savefig()` → output to `figures/`
- Requires at least one PNG output

### DataQualityValidator

Validates data quality automatically:

```python
validator = DataQualityValidator(df, num_features, cat_features)
validator.validate()
# Prints report and offers auto-fixes
```

**Checks**:
- Synthetic patterns (uniform distributions)
- Outliers (IQR method)
- Domain constraints (ages, percentages, times)
- Temporal logic (sequence integrity)

### AI Agents

#### DecisionMaker
- Uses OpenAI Chat API
- Plans next analysis steps
- Synthesizes final report
- Maintains global analysis context

#### CodexGenerator
- Uses OpenAI Responses API
- Generates Python analysis code
- Handles NaNs and data types
- Creates multiple plots

#### ReActAnalyzer
- Orchestrates entire loop
- Maintains state and observations
- Calls DecisionMaker → CodexGenerator → SafeRunner
- Handles error recovery

---

## 🐛 Troubleshooting

### Issue: `OPENAI_API_KEY not found`

**Solution**:
```bash
# Set environment variable
export OPENAI_API_KEY="sk-your-key"

# Or pass via CLI
python AgentResearchAssitant.py --api-key "sk-your-key" --input data.csv
```

### Issue: `FileNotFoundError: [Errno 2] No such file or directory: 'pandoc'`

**Solution**: Install pandoc
```bash
# macOS
brew install pandoc

# Windows (download from https://pandoc.org/install.html)
# Linux
sudo apt-get install pandoc
```

### Issue: `ModuleNotFoundError: No module named 'openai'`

**Solution**:
```bash
# Verify virtual environment is activated
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: `Rate limit exceeded` (OpenAI API)

**Solution**:
- Reduce `--max-steps`
- Use a cheaper model (gpt-4o-mini instead of gpt-4)
- Wait and retry after some time
- Check usage at https://platform.openai.com/account/usage

### Issue: `KeyError: 'figures_dir'` or figures not saving

**Solution**:
- Ensure `figures/` directory exists in output folder
- Check write permissions on output directory
- Verify plot code uses `plt.savefig()` (SafeRunner will rewrite it)

---

## 📝 Hackathon Notes (Tecnun-IAthon)

This repository is a **starting point for experimentation**:

### ✅ You Are Encouraged to:
- Modify prompting strategies for better analysis
- Add new agents (e.g., HypothesisTester, DashboardBuilder)
- Extend validation checks and auto-fix logic
- Customize report structure and templates
- Improve cost control and budgeting

### 💡 Tips:
- Use `gpt-4o-mini` or smaller models to control costs
- Check `token_usage.log` after each run
- Be mindful of token usage (especially with large datasets)
- Test with small datasets first
- Customize the system prompts for your use case

### 🎯 Example Extensions:
```python
# Add a custom agent
class DashboardBuilder(Agent):
    def build_interactive_dashboard(self, df):
        # Create Streamlit/Plotly interactive dashboard
        pass

# Integrate into ReActAnalyzer
analyzer.add_agent("dashboard", DashboardBuilder())
```

---

## 📚 Additional Files

- **AgentResearchAssitant.py** - Main entry point (the ResearchAssistant)
- **md_to_ppt.py** - Converts Markdown to PowerPoint (decreases complexity)
- **RetrieveImage.py** - Utility to download images from internet by description

---

## 🔍 Quick Links

- [GitHub Repository](https://github.com/arubio2/CleanCode)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pandoc Documentation](https://pandoc.org/)
- [Issue Tracker](https://github.com/arubio2/CleanCode/issues)

---

**Last Updated**: January 2026  
**Project**: Tecnun-IAthon ResearchAssistant  
**Status**: Active Development  
**Language**: Python 100%
