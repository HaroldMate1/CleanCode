#!/usr/bin/env python3
"""
IAthon Agentic Starter Kit - AI-Driven Data Analysis Report Generator
======================================================================

This script uses an agentic AI approach where the LLM writes and executes
Python code to analyze data dynamically, rather than following a fixed pipeline.

Usage:
    python app.py --input data.csv --output report.md [--format pptx|docx|pdf] [--api-key YOUR_KEY]

Requirements:
    pip install openai pandas plotly scipy numpy openpyxl kaleido
"""
import argparse
import json
import os
import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import pandas as pd
import numpy as np
from openai import OpenAI
import plotly.io as pio
from scipy import stats
import time

def universal_cleaner(df, nan_threshold=0.5, drop_constant=True):
    start_time = time.time()
    initial_cols = df.columns.tolist()
    
    print("--- STARTING DATA CLEANING LOG ---")
    print(f"Initial Shape: {df.shape}")

    # 1. Drop high-NaN columns
    limit = len(df) * (1 - nan_threshold)
    df_dropped_nas = df.dropna(thresh=limit, axis=1)
    dropped_nas = set(initial_cols) - set(df_dropped_nas.columns)
    if dropped_nas:
        print(f"DROPPED (High NaNs > {nan_threshold*100}%): {list(dropped_nas)}")
    
    # 2. Drop constant columns (zero variance)
    if drop_constant:
        constant_cols = [col for col in df_dropped_nas.columns if df_dropped_nas[col].nunique() <= 1]
        df_clean = df_dropped_nas.drop(columns=constant_cols)
        if constant_cols:
            print(f"DROPPED (Constant/Zero Variance): {constant_cols}")
    else:
        df_clean = df_dropped_nas

    # 3. Sanitize Strings & Fix Symbols
    str_cols = df_clean.select_dtypes(include=['object']).columns
    replacements_made = 0
    for col in str_cols:
        # Check if '$' exists before replacing to avoid unnecessary work
        if df_clean[col].dtype == 'object' and df_clean[col].str.contains(r'\$', na=False).any():
            # Also ensure the replace call uses the correct flags
            df_clean[col] = df_clean[col].str.replace(r'$', 's', regex=False)
            replacements_made += 1
    
    if replacements_made > 0:
        print(f"CLEANED: Special characters ($) fixed in {replacements_made} text columns.")

    # 4. Auto-detect Feature Types
    numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    end_time = time.time()
    print(f"\nFinal Shape: {df_clean.shape}")
    print(f"Process took: {end_time - start_time:.2f} seconds")
    print("--- END OF LOG ---\n")
    return df_clean, numeric_features, categorical_features

class DataPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def run(self):
        # 1. Load the data
        raw_df = pd.read_csv(self.file_path)
        
        # 2. CALL THE FUNCTION HERE
        # This is where the 'universal_cleaner' is actually used
        self.data, self.num_features, self.cat_features = universal_cleaner(raw_df)
        
        print("Cleaning finished. Data is ready for EDA.")


class SafeCodeRunner:
    FORBIDDEN_IMPORTS = ['os.remove', 'os.unlink', 'shutil.rmtree', 'subprocess', 'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib', 'telnetlib', '__import__', 'eval', 'exec', 'compile', 'open']
    FORBIDDEN_PATTERNS = [r'os\.system', r'os\.popen', r'globals\(\)', r'locals\(\)', r'vars\(\)', r'delattr', r'setattr', r'__file__', r'__builtins__']

    def __init__(self, data_path, output_dir, verbose=False):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        self.execution_count = 0
        self.max_executions = 50
        self.persistent_vars = {}
        self._load_initial_data()

    def _load_initial_data(self):
            """Loads and automatically cleans the data using the universal_cleaner."""
            try:
                print(f"📥 Loading raw data: {self.data_path}")
                ext = self.data_path.suffix.lower()
                if ext == '.csv': 
                    df = pd.read_csv(self.data_path)
                elif ext in ['.xlsx', '.xls']: 
                    df = pd.read_excel(self.data_path)
                else: 
                    df = pd.read_json(self.data_path)

                # --- INTEGRATION OF UNIVERSAL CLEANER ---
                # We call the cleaner here so it happens once at the start.
                # This uses the 'df' loaded from the CLI argument (self.data_path).
                df_clean, num_cols, cat_cols = universal_cleaner(df)
                
                # Store the cleaned results in the persistent variables for the AI
                self.persistent_vars['df'] = df_clean
                self.persistent_vars['num_features'] = num_cols
                self.persistent_vars['cat_features'] = cat_cols
                
                if self.verbose: 
                    print(f"✅ Data cleaned successfully. Shape: {df_clean.shape}")
                    
            except Exception as e: 
                print(f"❌ Error loading/cleaning data: {e}")

    def execute_code(self, code, description="Code execution"):
        if self.execution_count >= self.max_executions: return {'success': False, 'error': 'Limit reached'}
        self.execution_count += 1
        
        # 1. SILENCE MATPLOTLIB (Set non-interactive backend)
        try:
            import matplotlib
            matplotlib.use('Agg') 
            import matplotlib.pyplot as plt
            plt.show = lambda *args, **kwargs: None # Dummy function
        except ImportError:
            pass

        # 2. SILENCE PLOTLY (Monkey-patch the show method)
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            plt.show = lambda *args, **kwargs: None
        except ImportError:
            sns = None

        safe_globals = {
            'pd': pd, 'np': np, 'px': px, 'go': go, 'pio': pio,
            'sns': sns, # Provide seaborn
            'plt': plt,
            'display': print, # Redirect display() to print()
            'stats': stats, 'Path': Path,
            'df': self.persistent_vars.get('df'),
            'FIGURES_DIR': str(self.output_dir / 'figures')
        }
        # Monkey-patch plt.show again inside the sandbox just in case
        plt.show = lambda *args, **kwargs: None
        safe_globals.update(self.persistent_vars)
        
        import io
        from contextlib import redirect_stdout
        output_buffer = io.StringIO()
        
        try:
            with redirect_stdout(output_buffer):
                exec(code, safe_globals)
            for k, v in safe_globals.items():
                if not k.startswith('_') and k not in ['pd', 'np', 'px', 'go', 'stats', 'pio', 'sns', 'plt', 'display']:
                    self.persistent_vars[k] = v
            
            printed = output_buffer.getvalue()
            if self.verbose and printed: print(f"DEBUG Output:\n{printed}")
            return {'success': True, 'printed': printed, 'code': code}
        except Exception as e:
            if self.verbose: print(f"DEBUG Error: {str(e)}")
            return {'success': False, 'error': str(e)}

class CodexAnalyzer:
    def __init__(self, api_key=None, model="gpt-5.1-codex-max", verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.verbose = verbose

    def generate_code(self, task, runner):
        num_cols = runner.persistent_vars.get("num_features", [])
        cat_cols = runner.persistent_vars.get("cat_features", [])

        prompt = f"""
                You are an expert data scientist.

                AVAILABLE OBJECTS:
                - df: pandas DataFrame
                - num_features: {num_cols}
                - cat_features: {cat_cols}
                - FIGURES_DIR: valid directory path (string)

                LIBRARIES:
                - pandas as pd
                - numpy as np
                - plotly.express as px
                - plotly.graph_objects as go
                - seaborn as sns
                - matplotlib.pyplot as plt
                - scipy.stats as stats

                STRICT RULES:
                - OUTPUT PYTHON CODE ONLY
                - NO markdown, NO backticks
                - NO explanations
                - NEVER call fig.show() or plt.show()
                - ALWAYS save figures to FIGURES_DIR
                - Use high DPI (>=300) for matplotlib
                - Prefer Plotly when possible
                - Use statistically appropriate tests

                TASK:
                {task}
                """
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )

        code = response.output_text.strip()

        if self.verbose:
            print("🧠 Codex code preview:\n", code[:500])

        return code

class AgenticAnalyzer:
    def __init__(self, api_key=None, model='gpt-5.1', verbose=False):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.verbose = verbose
        self.conversation_history = []
        
    def _call_llm(self, prompt, system_message=None):
        messages = []
        if system_message: messages.append({"role": "system", "content": system_message})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=1)
        assistant_message = response.choices[0].message.content
        if self.verbose: print(f"\n🧠 DEBUG AI Thought: {assistant_message[:300]}...\n")
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    
    def extract_code_blocks(self, text):
        return re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    
    def analyze_with_code_execution(self, runner, initial_user_prompt, max_iterations=10):
        # Fetch the features we identified during cleaning
        num_cols = runner.persistent_vars.get('num_features', [])
        cat_cols = runner.persistent_vars.get('cat_features', [])
        
        system_message = f"""
            You are an expert data scientist.
            You are writing a formal executive report. Do not use 'I' or describe your thought process. State findings as objective facts derived from the data. Use markdown headers for organization.
            DATA CONTEXT:
            - 'df': Main DataFrame
            - 'num_features': {num_cols}
            - 'cat_features': {cat_cols}
            ENVIRONMENT CONTEXT:
            - Libraries available: pandas (pd), numpy (np), plotly (px, go), seaborn (sns), matplotlib.pyplot (plt), scipy.stats.
            - LIBRARIES NOT AVAILABLE: missingno, ydata_profiling.
            - Use print() to output results.
            - 'df' is already loaded.
            STRICT RULES:
            - Use f-strings for paths: plt.savefig(f"{{FIGURES_DIR}}/my_plot.png")
            - NEVER write 'FIGURES_DIR/plot.png' as a literal string. 
            - The variable FIGURES_DIR is a string path already provided in your environment.
            - NEVER use fig.show() or plt.show(). This is a headless environment.
            - ALWAYS save figures to FIGURES_DIR using fig.write_image('path.png') or fig.write_html('path.html').
            - If you use Matplotlib, save with plt.savefig('path.png').
            GOAL: Generate a 3,000-word report. 
            STRUCTURE: Intro, Data Representation (Plotly/Maps), Interesting Facts (Stats tests), Discussion (Correlation vs Causality), Conclusion.
            """
        initial_prompt = initial_user_prompt
        analysis_log = []
        
        for iteration in range(1, max_iterations + 1):
            prompt = initial_prompt if iteration == 1 else "Continue analysis. Use printed output to decide next steps."
            response = self._call_llm(prompt, system_message=system_message)

            if "ANALYSIS_COMPLETE" in response: break
            
            blocks = self.extract_code_blocks(response)
            for code in blocks:
                result = runner.execute_code(code, f"Iteration {iteration}")
                analysis_log.append({'type': 'execution', 'code': code, 'result': result})
    
                if result['success']:
                    feedback = f"Result Output:\n{result.get('printed', 'Executed successfully.')}"
                else:
                    feedback = f"EXECUTION ERROR: {result.get('error')}\nPlease fix the code and try again."
                self.conversation_history.append({"role": "user", "content": feedback})     
        return analysis_log

class VisualizationEnhancer:
    def __init__(self, api_key=None, model='gpt-5.1'):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def enhance_figure(self, runner, finding_text):
        prompt = f"Create a Plotly chart for: '{finding_text}'. Save as PNG/HTML in FIGURES_DIR. NO fig.show()."
        response = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        code = re.findall(r'```python\n(.*?)```', response.choices[0].message.content, re.DOTALL)
        if code: return runner.execute_code(code[0], "Enhanced Viz")
        return None

class ReportWriter:
    def __init__(self, api_key=None, model='gpt-5.1'):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call(self, prompt):
        res = self.client.chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content

    def write_introduction(self, data_info):
        return self._call(f"Write a professional introduction for a report on: {data_info}")

    def write_discussion(self, findings):
        return self._call(f"Analyze and discuss these findings: {findings}")

    def write_conclusion(self, findings):
        return self._call(f"Summarize key takeaways: {findings}")

def generate_caption_from_filename(filename):
    name = filename.stem.replace("_", " ")
    return (
        f"{name.capitalize()}. "
        "Visualization generated automatically during exploratory analysis. "
        "See main text for statistical interpretation."
    )

class AgenticReportGenerator:
    def __init__(self, data_path, output_path, api_key=None, verbose=False):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_dir = self.output_path.parent
        self.verbose = verbose
        self.runner = SafeCodeRunner(self.data_path, self.output_dir, verbose=verbose)
        self.analyzer = AgenticAnalyzer(api_key=api_key, verbose=verbose)
        self.viz_enhancer = VisualizationEnhancer(api_key=api_key)
        self.writer = ReportWriter(api_key=api_key)

    def generate_report(self, user_prompt):
        print("🤖 Analyzing data (Heads-down mode)...")
        analysis_log = self.run_coding_phase()
        print("🎨 Enhancing visualizations...")
        self._enhance_visualizations(analysis_log)
        print("✍️ Writing final report...")
        report = self._write_final_report(analysis_log)
        self.output_path.write_text(report, encoding='utf-8')
        return report

    def _write_final_report(self, analysis_log):
        df = self.runner.persistent_vars.get('df')
        summary = f"Dataset with {df.shape[0]} rows. Columns: {list(df.columns)}"
        
        findings = "\n---\n".join([
            f"Execution Output:\n{e['result'].get('printed', '')}" 
            for e in analysis_log if e['type'] == 'execution' and e['result']['success']
        ])
        
        intro = self.writer.write_introduction(summary)
        # The writer now has the ACTUAL data to discuss
        disc = self.writer.write_discussion(findings) 
        conc = self.writer.write_conclusion(findings)

        viz_md = self._format_visualizations()

        return f"# Data Analysis Report\n\n{intro}\n\n## Visual Analysis\n{viz_md}\n\n## Discussion\n{disc}\n\n## Conclusion\n{conc}"

    def _enhance_visualizations(self, analysis_log):
        results_to_visualize = [
        e['result'].get('printed', '') 
        for e in analysis_log if e['type'] == 'execution' and e['result']['success']
        ]
    
        for text in results_to_visualize[:3]:
            if len(text.strip()) > 20: # Only visualize meaningful output
                self.viz_enhancer.enhance_figure(self.runner, text)

    def _format_visualizations(self):
        figs = sorted((self.output_dir / 'figures').glob('*.png'))
        md = []
        for i, f in enumerate(figs, 1):
            caption = generate_caption_from_filename(f)
            md.append(f"![Figure {i}. {caption}](figures/{f.name})")
        return "\n\n".join(md)

    def convert_to_format(self, fmt):
        out = self.output_path.with_suffix(f'.{fmt}')
        print(f"📄 Converting to {fmt.upper()}...")
        try:
            subprocess.run(['pandoc', str(self.output_path), '-o', str(out)], check=True)
            print(f"✅ Saved: {out}")
        except Exception as e: print(f"❌ Error: {e}")
    
    def run_coding_phase(self):
        print("🧪 Running statistical analysis & figures (Codex)...")

        codex = CodexAnalyzer(
            api_key=self.analyzer.api_key,
            verbose=self.verbose
        )

        tasks = [
            "Perform exploratory data analysis on numeric variables. "
            "Generate at least 3 insightful visualizations.",

            "Test for correlations between numeric variables. "
            "Use Pearson or Spearman appropriately and report p-values.",

            "Compare distributions across categorical variables "
            "using appropriate statistical tests.",

            "Create one summary visualization combining multiple variables."
        ]

        execution_log = []

        for task in tasks:
            code = codex.generate_code(task, self.runner)
            result = self.runner.execute_code(code, description=task)
            execution_log.append(result)

        return execution_log

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--format', '-f', choices=['pdf', 'docx', 'pptx'])
    parser.add_argument('--api-key')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    generator = AgenticReportGenerator(args.input, args.output, api_key=args.api_key, verbose=args.verbose)
    
    # Define the IAthon requirements here
    prompt = """
    Analyze the dataset for IAthon. 
    1. Provide a 3,000-word Markdown report.
    2. Use Plotly for all visuals (scatter/heatmaps).
    3. Conduct statistical tests to verify facts.
    4. Discuss results critically (correlation != causality).
    """
    generator.generate_report(prompt)
    if args.format:
        generator.convert_to_format(args.format)

if __name__ == '__main__':
    main()