# CleanCode - AI Research Assistant

An AI-powered data analysis tool that automatically explores your data, generates visualizations, runs statistical tests, and produces professional reports.

**Give it a CSV/Excel file → Get a comprehensive report with insights and visualizations**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## IAthon Challenge

This research agent is your starting point for the IAthon! Your challenge is to:

1. **Run the agent** on sample datasets to generate PDF reports, PowerPoint presentations, and visualizations
2. **Analyze the output** - What's good? What's missing? What could be better?
3. **Improve the agent** - Enhance the code, add new features, improve visualizations, or refine the analysis
4. **Show your improvements** - The better your improvements, the better your IAthon outcome!

Think of this as a hackathon where you're not starting from scratch. You have a working AI research assistant that already does data analysis, statistical testing, and report generation. Your job is to make it even better!

**Possible improvements:**
- Better visualizations or new chart types
- More sophisticated statistical tests
- Improved report formatting
- Better error handling
- New export formats
- Enhanced AI prompts for deeper insights
- Performance optimizations
- Custom analysis pipelines

---

## What It Does

- **Validates & cleans** your data automatically
- **Generates visualizations** (distributions, correlations, comparisons)
- **Runs statistical tests** (t-tests, ANOVA, chi-square, etc.)
- **Produces reports** in Markdown, PDF, DOCX, or PowerPoint
- **Tracks costs** for OpenAI API usage

---

## Quick Start

### Prerequisites

- **Python 3.8+**
- **OpenAI API Key** (will be provided during the IATHON session)
- **Pandoc** (for PDF/DOCX/PPTX conversion)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/arubio2/CleanCode.git
cd CleanCode
```

#### 2. Create a Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install Pandoc (Optional)

**Required only if you want PDF/DOCX/PPTX output.**

**macOS (using Homebrew):**
```bash
brew install pandoc
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install pandoc
```

**Windows:**

Choose one of these methods:

1. **Direct Download (Recommended):**
   - Visit [pandoc.org/installing.html](https://pandoc.org/installing.html)
   - Download the Windows installer (.msi file)
   - Run the installer
   - Verify installation: `pandoc --version`

2. **Using winget (Windows 10 1809+) (not recommended):**
   ```powershell
   winget install --id JohnMacFarlane.Pandoc
   ```


---

## Configuration

### Set Your OpenAI API Key

> **📢 For IAthon Participants:**
> An OpenAI API key will be **provided to you during the IAthon session**.
> Once you receive it, follow the instructions below to configure it.

**Option 1: Environment Variable (Recommended for Hackathon)**

**macOS/Linux:**
```bash
# Replace 'your-actual-key-from-IAthon' with the key provided during the session
export OPENAI_API_KEY="sk-proj-your-actual-key-from-IAthon"
```

**Windows (Command Prompt):**
```cmd
# Replace with the key provided during the IAthon session
$env:OPENAI_API_KEY="sk-proj-your-actual-key-from-IAckathon"
```


## Usage

### Basic Usage

Analyze a CSV file and generate a Markdown report:

```powershell
python AgentResearchAssitant.py --input DataSets/yourfile.csv --output results/report.md
```

### Generate PowerPoint Report

```powershell
python AgentResearchAssitant.py --input DataSets/yourfile.csv --output results/report.md --format pptx
```

### Generate PDF Report

```powershell
python AgentResearchAssitant.py --input DataSets/yourfile.csv --output results/report.md --format pdf
```

### Advanced Options

```powershell
python AgentResearchAssitant.py --input DataSets/yourfile.csv --output results/report.md --format pptx --max-steps 7 --verbose --prompt "Focus on gender disparities and test score correlations"
```

### Custom Analysis Prompt

Use a text file for longer prompts:

```powershell
python AgentResearchAssitant.py --input data.csv --output results/report.md --prompt-file my_analysis_requirements.txt
```

---

## Command-Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Path to CSV or Excel file |
| `--output` | `-o` | Yes | Output path for report (`.md`) |
| `--format` | `-f` | No | Output format: `pptx`, `pdf`, or `docx` |
| `--api-key` | | No | OpenAI API key (overrides environment variable) |
| `--max-steps` | | No | Max analysis iterations (default: 5) |
| `--verbose` | `-v` | No | Enable detailed logging |
| `--prompt` | `-p` | No | Custom analysis requirements |
| `--prompt-file` | | No | Load custom prompt from file |

---

## Output Structure

After running the tool, you'll get:

```
results/
├── report.md              # Markdown report with analysis
├── report.pptx           # PowerPoint (if --format pptx)
├── figures/              # Generated visualizations
│   ├── distribution_001.png
│   ├── correlation_002.png
│   └── ...
└── token_usage.log       # OpenAI API cost breakdown
```

---

## Example Datasets

The repository includes 7 sample datasets in the `DataSets/` folder:

- `StudentsPerformance.csv` - Student test scores with demographics
- `Salary.csv` - Employment and salary data
- `2019 Ironman World Championship Results.csv` - Athletic performance data
- `olympic-results.csv` - Olympic competition results
- `jamb_exam_results.csv` - Exam results data
- `support2.csv` - Customer support tickets
- `world_bank_data_2025.csv` - World Bank indicators

Try them out:

```bash
python AgentResearchAssitant.py --input DataSets/StudentsPerformance.csv --output results/students.md --format pptx
```

---

## Features

### Data Quality Validation

Automatically detects and optionally fixes:
- Synthetic or unrealistic patterns
- Statistical outliers
- Invalid domain values (negative ages, impossible percentages)
- Temporal logic errors (death before birth)

### Safe Code Execution

AI-generated analysis code runs in a sandboxed environment:
- Prevents file system access
- Blocks dangerous operations
- Validates all outputs

### Cost Tracking

Tracks OpenAI API usage:
- Fetches live pricing (with caching)
- Shows cost per model
- Saves detailed breakdown to `token_usage.log`

### Multi-Format Reports

Export to:
- **Markdown** - Plain text with embedded images
- **PDF** - Professional document (requires Pandoc)
- **DOCX** - Microsoft Word (requires Pandoc)
- **PPTX** - PowerPoint presentation (requires Pandoc)

---

## How It Works

1. **Loads & validates** your data
2. **AI agents plan** what analyses to run
3. **Generates Python code** for visualizations and statistics
4. **Executes safely** in a sandboxed environment
5. **Synthesizes findings** into a professional report
6. **Converts** to your preferred format

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"

Make sure you've activated the virtual environment and installed dependencies:

```bash
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### "OpenAI API key not found"

Set the `OPENAI_API_KEY` environment variable or pass it via `--api-key`:

```bash
export OPENAI_API_KEY="sk-proj-your-key"  # macOS/Linux
set OPENAI_API_KEY=sk-proj-your-key      # Windows CMD
```

### "pandoc: command not found"

Install Pandoc from [pandoc.org](https://pandoc.org/installing.html) or skip PDF/DOCX/PPTX conversion by omitting `--format`.

### "Permission denied" errors on Windows

If you see permission errors when running the script, try:
1. Run terminal as Administrator
2. Disable antivirus temporarily
3. Use `--verbose` flag to see detailed error messages

### API Rate Limits

If you hit OpenAI rate limits:
- Reduce `--max-steps` (try 3-4 instead of 5)
- Use smaller models in the code (edit model names in `AgentResearchAssitant.py`)
- Wait a few minutes and retry

---

## Cost Estimates

Typical analysis costs (using GPT-4 models):
- Small dataset (<1000 rows): $0.10 - $0.50
- Medium dataset (1000-10000 rows): $0.50 - $2.00
- Large dataset (>10000 rows): $2.00 - $5.00

Costs vary based on:
- Dataset complexity
- Number of features
- Analysis depth (`--max-steps`)
- Model selection

Check `token_usage.log` after each run for exact costs.

---

## Advanced Usage

### Standalone Markdown to PowerPoint

Convert an existing Markdown report to PowerPoint:

```bash
python md_to_ppt.py \
  --md results/report.md \
  --pptx template.pptx \
  --api_key $OPENAI_API_KEY \
  --output results/presentation.pptx
```

### Download Illustrative Images

Use `RetrieveImage.py` to download contextual images from Pexels:

```python
from RetrieveImage import download_image

download_image("data analysis", "your_pexels_api_key", "analysis.jpg")
```

---

## For Developers

### Project Structure

```
CleanCode/
├── AgentResearchAssitant.py   # Main analysis pipeline
├── md_to_ppt.py                # Markdown to PowerPoint converter
├── RetrieveImage.py            # Image download utility
├── requirements.txt            # Python dependencies
├── DataSets/                   # Sample datasets
├── README.md                   # This file
└── ARCHITECTURE.md             # Detailed technical documentation
```

### Extending the Tool

This project is designed for customization:
- Modify AI prompts in the agent classes
- Add new validation checks in `DataQualityValidator`
- Create custom report templates
- Add new analysis types

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation.

---

## Contributing

Contributions welcome! This project was created for the Tecnun-IAthon hackathon.

Ideas for contributions:
- Additional data validation rules
- New visualization types
- Support for more file formats
- Custom report templates
- Better error handling

---

## License

MIT License - See LICENSE file for details.

---

## Credits

Created for the Tecnun-IAthon hackathon.

Repository: [github.com/arubio2/CleanCode](https://github.com/arubio2/CleanCode)

---

## Need Help?

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
- Review sample datasets in `DataSets/` folder
- Open an issue on GitHub
- Review `token_usage.log` for cost analysis

**Happy analyzing!** 📊🤖
