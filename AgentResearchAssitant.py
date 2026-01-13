#!/usr/bin/env python3
"""
IAthon – Iteration 6.5 (Robust Context Version)
Fixes: Windows/OneDrive PermissionError during folder cleanup.
Features: Enhanced domain discovery and image embedding.
"""

import argparse
import os
import shutil      
import subprocess
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- Core scientific stack ---
import pandas as pd
import numpy as np
from scipy import stats

# --- Plotting (headless-safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px
import plotly.graph_objects as go

from openai import OpenAI

# ======================================================
# Utilities
# ======================================================
def get_api_key(cli_key=None):
    return cli_key or os.getenv("OPENAI_API_KEY")

def robust_rmtree(path: Path):
    """Attempt to delete a directory, handling Windows/OneDrive locks."""
    if not path.exists():
        return
    for _ in range(3): # Try 3 times
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1) # Wait for sync/lock to release
    # Fallback: Just clear files inside instead of deleting folder
    for item in path.iterdir():
        try:
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)
        except:
            pass

# ======================================================
# 1. DATA LOADING & CLEANING
# ======================================================
def load_and_clean(path: Path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    df = df.loc[:, df.nunique() > 1]

    num_features = df.select_dtypes(include=np.number).columns.tolist()
    cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return df, num_features, cat_features

# ======================================================
# 2. SAFE EXECUTION ENVIRONMENT
# ======================================================
class SafeRunner:
    def __init__(self, df, num_features, cat_features, output_dir: Path, verbose=False):
        self.df = df
        self.num_features = num_features
        self.cat_features = cat_features
        self.output_dir = output_dir.resolve() 
        self.verbose = verbose
        self.figures_dir = self.output_dir / "figures"
        
        if self.figures_dir.exists():
            if self.verbose: print(f"🧹 Cleaning old figures (Robust mode)...")
            robust_rmtree(self.figures_dir)
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, code):
        if "```" in code:
            match = re.search(r"```(?:python)?\n?(.*?)\n?```", code, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                code = "\n".join([line for line in code.split("\n") if "```" not in line])

        globals_safe = {
            "df": self.df, "num_features": self.num_features, "cat_features": self.cat_features, 
            "plt": plt, "sns": sns, "px": px, "go": go, "stats": stats, "np": np, "pd": pd,
            "FIGURES_DIR": str(self.figures_dir) 
        }
        
        old_cwd = os.getcwd()
        os.chdir(self.figures_dir) 
        try:
            exec(code, globals_safe)
        finally:
            os.chdir(old_cwd)

# ======================================================
# 3. DECISION MAKER (CHAT MODEL)
# ======================================================
class DecisionMaker:
    def __init__(self, api_key, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.verbose = verbose
    
    def synthesize_report(self, analysis_log, original_prompt, figures_dir, data_preview):
        if self.verbose: print("✍️ Synthesizing domain-aware report...")
        
        figs = sorted(figures_dir.glob("*.png"))
        fig_list = "\n".join([f"- {f.name}" for f in figs])

        synthesis_prompt = f"""
        You are a world-class Data Scientist and Technical Writer.
        
        DATA PREVIEW (Use this to discover the context):
        {data_preview}

        ANALYSIS HISTORY:
        {analysis_log}

        AVAILABLE FIGURES:
        {fig_list}

        YOUR TASK:
        1. CONTEXT IDENTIFICATION: Based on the preview, what is this data? (e.g., Ironman Race results).
        2. DOMAIN VOCABULARY: Use the correct terminology. For races, use 'splits', 'pacing', 'transitions (T1/T2)', and 'overall time'. 
        3. REPORT STRUCTURE: Intro, Detailed Results (with images), Statistical Discussion, and Conclusion.
        4. IMAGE EMBEDDING: Place ![Description](figures/filename.png) immediately after the text that discusses it.
        
        STYLE: High-impact scientific paper.
        """
        response = self.client.chat.completions.create(
            model="gpt-5.1", 
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.7 
        )
        return response.choices[0].message.content

    def decide(self, observations: str, step_num: int, max_steps: int, 
           num_features: list, cat_features: list, data_preview: str) -> str:
        
        prompt = f"""Analyze the DATA PREVIEW below to identify the domain.
                
                DATA PREVIEW:
                {data_preview}

                STEP: {step_num + 1} of {max_steps}
                CURRENT OBSERVATIONS: {observations}

                MISSION: 
                - Deduce the origin/subject of the data.
                - Propose a specific visualization or statistical test.
                - If columns represent times (Split/Swim/Bike), suggest converting them to seconds.
                
                DECISION (ONE ACTION):"""
        
        response = self.client.chat.completions.create(
            model="gpt-5.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

# ======================================================
# 4. CODE GENERATOR (CODEX)
# ======================================================
class CodexGenerator:
    def __init__(self, api_key, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.verbose = verbose

    def generate(self, instruction, num_features, cat_features, data_preview) -> str:
        prompt = f"""
            You are an expert data scientist.
            DATA PREVIEW: {data_preview}
            TASK: {instruction}
            
            RULES:
            - If columns contain time strings (e.g., HH:MM:SS), you MUST use pd.to_timedelta() and .dt.total_seconds() before plotting.
            - OUTPUT PYTHON CODE ONLY. NO Markdown.
            - ALWAYS save to FIGURES_DIR.
            """
        response = self.client.responses.create(
            model="gpt-5.1-codex-max",
            input=prompt,
        )
        return response.output_text.strip()

# ======================================================
# 5. REACT ORCHESTRATOR
# ======================================================
class ReActAnalyzer:
    def __init__(self, runner, decider, coder, max_steps=20, verbose=False):
        self.runner = runner
        self.decider = decider
        self.coder = coder
        self.max_steps = max_steps
        self.verbose = verbose
        self.observations = []
        self.analysis_log = []
        self.data_preview = self.runner.df.head(10).to_string() # Increased preview for better context

    def observe(self):
        figs = sorted(self.runner.figures_dir.glob("*.png"))
        obs = f"- Figures: {len(figs)}\n- Files: " + ", ".join([f.name for f in figs]) if figs else "- No figs."
        self.observations.append(obs)

    def run(self, user_requirements): 
        print(f"\n🔍 STARTING ANALYSIS (Iteration 6.5)\n{'='*70}")
        self.observe()
        
        for step in range(self.max_steps):
            print(f"\n📍 STEP {step + 1}/{self.max_steps}")
            decision = self.decider.decide("\n".join(self.observations), step, self.max_steps, self.runner.num_features, self.runner.cat_features, self.data_preview)
            
            if "STOP" in decision.upper(): break
            
            code = self.coder.generate(decision, self.runner.num_features, self.runner.cat_features, self.data_preview)
            
            try:
                self.runner.run(code)
                if self.verbose: print("⚡ Success.")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            self.observe()
            self.analysis_log.append({"step": step + 1, "thought": decision, "result": self.observations[-1]})

        report_text = self.decider.synthesize_report(self.analysis_log, user_requirements, self.runner.figures_dir, self.data_preview)
        report_path = self.runner.output_dir / "report.md" 
        report_path.write_text(report_text, encoding='utf-8')
        print(f"✨ Analysis complete. Report saved to {report_path}")
        return report_path

# ======================================================
# 6. CLI
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-f", "--format", choices=["pdf", "docx", "pptx"])
    parser.add_argument("--api-key", required=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key(args.api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_dir = output_path.parent

    df, num_features, cat_features = load_and_clean(input_path)
    runner = SafeRunner(df, num_features, cat_features, output_dir, verbose=args.verbose)
    decider = DecisionMaker(api_key, verbose=args.verbose)
    coder = CodexGenerator(api_key, verbose=args.verbose)

    analyzer = ReActAnalyzer(runner, decider, coder, max_steps=20, verbose=args.verbose)
    report_md_path = analyzer.run("Comprehensive domain-specific report with embedded figures.") 

    if args.format:
        print(f"📦 Converting to {args.format}...")
        out_file = output_path.with_suffix(f".{args.format}")
        subprocess.run(["pandoc", report_md_path.name, "-o", out_file.name], cwd=output_dir)

if __name__ == "__main__":
    main()