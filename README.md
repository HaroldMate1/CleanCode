
### Big-picture explanation

At a high level, the program does this:

1. **CLI + Setup**
   - Reads command-line arguments (input file, output path, format, custom prompt, API key).
   - Creates an output directory.
   - Initializes a **UsageTracker** that:
     - Fetches OpenAI pricing (live web scrape → cache → fallback table).
     - Tracks tokens/cost across models.

2. **Data Loading + Validation**
   - `load_and_clean()` reads CSV/Excel into a DataFrame, drops bad columns, splits numeric vs categorical features.
   - **DataQualityValidator**:
     - Runs checks for synthetic patterns, outliers, domain constraints (ages, percentages, times), temporal logic.
     - Prints a human-readable summary (errors/warnings/info).
     - Optionally auto-fixes some issues (negative times, invalid ages).

3. **Safe Execution Environment**
   - **SafeRunner** holds the cleaned (or fixed) DataFrame.
   - It executes generated Python code with:
     - A safe global environment (df, plotting libraries, stats, np/pd, FIGURES_DIR).
     - Blocked file-read calls (no `pd.read_csv`, etc.).
     - Rewritten `plt.savefig(...)` calls so all plots land in an output `figures/` folder.
     - A requirement that at least one `.png` is saved, or it raises an error.

4. **AI “Agents”**
   - **DecisionMaker** (planning & reporting agent):
     - Uses OpenAI (chat/completions) to:
       - Plan batch analyses over several steps (ReAct-style loop).
       - Synthesize a final Markdown report that references the actual figures created.
     - Logs token usage via **UsageTracker**.
   - **CodexGenerator** (code-writing agent):
     - Uses OpenAI (responses API) to generate Python code blocks that:
       - Operate on the in-memory `df`.
       - Create multiple plots per batch and save them to `FIGURES_DIR`.
       - Obey robustness rules (check dtypes, handle NaNs, etc.).
     - Logs token usage via **UsageTracker**.

5. **ReActAnalyzer – Orchestrator**
   - **ReActAnalyzer** coordinates the loop:
     - Maintains observations, analysis log, error history, and a data preview.
     - For up to `max_steps`:
       - Asks **DecisionMaker** for the next “batch analysis” plan (or “STOP”).
       - Asks **CodexGenerator** to turn that plan (plus any recent errors) into Python code.
       - Runs that code with **SafeRunner**, then:
         - Records success/failure, updates observations and logs.
     - At the end, asks **DecisionMaker** to synthesize the final Markdown report.
     - Saves `report.md` in the output directory.

6. **Output Conversion + Cost Summary**
   - If requested (`--format`), uses `pandoc` to convert `report.md` to PDF/DOCX or PPTX, with:
     - Optional custom PPTX styling/template (if helper functions are available).
   - Finally, **UsageTracker**:
     - Prints a detailed cost summary by model.
     - Saves `token_usage.log` in the output directory.

---

```mermaid
flowchart TD
  %% Make the whole diagram about 50% larger
  classDef bigText font-size:18px;

  subgraph CLI["CLI and Setup"]
    A1[Parse args]
    A2[Resolve paths and output dir]
    A3[Get API key]
    A4[Init UsageTracker and pricing]
  end
  A1 --> A2 --> A3 --> A4

  subgraph DATA["Data Prep and Validation"]
    B1[load_and_clean data]
    B2[Init DataQualityValidator]
    B3[Run validate checks]
    B4{Validation passed?}
    B5[Ask user for auto fix]
    B6[Run auto_fix]
    B7[Use original df]
    B8[Use fixed df]
  end

  A4 --> B1 --> B2 --> B3 --> B4
  B4 -- "yes" --> B7
  B4 -- "no" --> B5
  B5 -- "y" --> B6 --> B8
  B5 -- "n" --> B7

  subgraph SAFE["SafeRunner"]
    C1[Init SafeRunner]
    C2[Prepare figures_dir]
    C3[Run user code safely]
  end

  B7 --> C1
  B8 --> C1
  C1 --> C2

  subgraph AGENTS["AI Agents"]
    D1[DecisionMaker]
    D2[decide next batch]
    D3[synthesize report]
    E1[CodexGenerator]
    E2[generate Python code]
  end

  C1 --> D1
  C1 --> E1
  D1 --> D2
  D1 --> D3
  E1 --> E2

  subgraph REACT["ReActAnalyzer"]
    F1[Init ReActAnalyzer]
    F2[Init state and preview]
    F3[observe figures]
    F4[run analysis loop]
  end

  C1 --> F1 --> F2
  F1 --> F3
  F3 --> F4

  subgraph OUTPUT["Reporting and Conversion"]
    G1[Main calls analyzer.run]
    G2{Output format}
    G3[Create pptx via pandoc]
    G4[Create pdf or docx via pandoc]
    G5[Keep markdown only]
    G6[Print usage summary]
    G7[Save usage log]
  end

  F4 --> G1
  G1 --> G2
  G2 -- "pptx" --> G3 --> G6
  G2 -- "pdf/docx" --> G4 --> G6
  G2 -- "none" --> G5 --> G6
  G6 --> G7

  subgraph TRACK["Token and Cost Tracking"]
    H1[UsageTracker state]
    H2[record text usage]
    H3[record image usage]
  end

  A4 --> H1
  D2 --> H2
  D3 --> H2
  E2 --> H2
  H1 --> G6
  H1 --> G7

  %% Apply the larger font to everything
  class CLI,DATA,SAFE,AGENTS,REACT,OUTPUT,TRACK,A1,A2,A3,A4,B1,B2,B3,B4,B5,B6,B7,B8,C1,C2,C3,D1,D2,D3,E1,E2,F1,F2,F3,F4,G1,G2,G3,G4,G5,G6,G7,H1,H2,H3 bigText;
```



### How to read this diagram

- **Top → bottom**: high-level pipeline: CLI → data loading & validation → safe runner → AI agents → ReAct loop → outputs & cost summary.
- **Subgraphs**:
  - **CLI**: how the tool is configured by the user.
  - **DATA**: how the DataFrame is cleaned and checked.
  - **SAFE**: sandbox for executing model-generated code.
  - **AGENTS**: Decider (plans + report) and Coder (code generation).
  - **REACT**: orchestrator that iteratively uses the agents + SafeRunner.
  - **OUTPUT**: report creation and optional pandoc conversion.
  - **TRACK**: central token/cost tracker used by all model calls.

## Illustration on how the three main agents interact

- **DecisionMaker**: decides *what* to do next (analysis step vs. code step vs. summarization).
- **CodexGenerator**: writes and updates Python code to run against the data.
- **ReActAnalyzer**: orchestrates the loop, feeding observations/results back into DecisionMaker and CodexGenerator until the analysis is done.

Below is a minimal flowchart showing their back‑and‑forth:

```mermaid
flowchart TD
  subgraph UserLoop["High-Level Agent Loop"]
    DM[DecisionMaker<br/>- Plans next action<br/>- Chooses: analyze, code, or summarize]
    CG[CodexGenerator<br/>- Writes/updates code<br/>- Calls SafeRunner]
    RA[ReActAnalyzer<br/>- Orchestrates loop<br/>- Tracks context & state]
  end

  %% Core interaction loop
  RA --> DM
  DM -- "Need new/updated code" --> CG
  CG -- "Code results, logs, figures" --> RA
  DM -- "High-level analysis / conclusions" --> RA

  %% Exit condition
  DM -- "Ready to finalize report" --> RA:::done

  classDef done fill:#c6f6d5,stroke:#2f855a,stroke-width:1px;
```

Key points in the diagram:

- **ReActAnalyzer** sits in the middle, repeatedly:
  - calling **DecisionMaker** to decide the next move,
  - asking **CodexGenerator** for new/updated code when needed,
  - aggregating outputs and deciding when the loop is complete.
- **DecisionMaker** is the strategist; **CodexGenerator** is the coder; **ReActAnalyzer** is the conductor.
