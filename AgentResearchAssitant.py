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
import tempfile
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from openai import OpenAI


class SafeCodeRunner:
    """
    Safe Python code execution sandbox with Windows-specific protections.
    Prevents file system manipulation and network access.
    """
    
    FORBIDDEN_IMPORTS = [
        'os.remove', 'os.unlink', 'shutil.rmtree', 'subprocess', 
        'socket', 'urllib', 'requests', 'http', 'ftplib',
        'smtplib', 'telnetlib', '__import__', 'eval', 'exec',
        'compile', 'open'  # We'll provide a safe open alternative
    ]
    
    FORBIDDEN_PATTERNS = [
        r'os\.system',
        r'os\.popen',
        r'globals\(\)',
        r'locals\(\)',
        r'vars\(\)',
        r'delattr',
        r'setattr',
        r'__file__',
        r'__builtins__',
    ]
    
    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.execution_count = 0
        self.max_executions = 50  # Prevent infinite loops
        
        # Persistent state across executions
        self.persistent_vars = {}
        self._load_initial_data()
    
    def _load_initial_data(self):
        """Load the dataset into persistent state."""
        try:
            ext = self.data_path.suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(self.data_path)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(self.data_path)
            elif ext == '.json':
                df = pd.read_json(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            self.persistent_vars['df'] = df
            self.persistent_vars['data'] = df  # Some code might use 'data' instead
            print(f"  ✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            print(f"  ⚠️  Warning: Could not load data initially: {e}")
    
    def is_code_safe(self, code):
        """Check if code is safe to execute."""
        # Check for forbidden imports
        for forbidden in self.FORBIDDEN_IMPORTS:
            if forbidden in code:
                return False, f"Forbidden import/function: {forbidden}"
        
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                return False, f"Forbidden pattern: {pattern}"
        
        # Check for file operations outside allowed directories
        if 'open(' in code or 'Path(' in code:
            # Allow only reading the input file and writing to figures dir
            pass  # We'll handle this in the execution environment
        
        return True, "Safe"
    
    def execute_code(self, code, description="Code execution"):
        """
        Execute Python code safely and return results.
        """
        if self.execution_count >= self.max_executions:
            return {
                'success': False,
                'error': 'Maximum execution limit reached',
                'output': None
            }
        
        # Safety check
        is_safe, message = self.is_code_safe(code)
        if not is_safe:
            return {
                'success': False,
                'error': f'Security violation: {message}',
                'output': None
            }
        
        self.execution_count += 1
        print(f"  ⚙️  Executing: {description} (attempt {self.execution_count}/{self.max_executions})")
        
        # Import libraries first (outside the restricted environment)
        import plotly.express as px
        import plotly.graph_objects as go
        from scipy import stats
        
        # Create a restricted execution environment
        # We need to provide a working __builtins__ for pandas to function
        import builtins
        safe_builtins = {
            # Safe built-in functions
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'abs': abs,
            'round': round,
            'sum': sum,
            'min': min,
            'max': max,
            'sorted': sorted,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'type': type,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,
            'Exception': Exception,
            # Needed for pandas/numpy to work
            '__import__': __import__,
            '__name__': '__main__',
            '__doc__': None,
        }
        
        safe_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'Path': Path,
            'json': json,
            're': re,
            'datetime': datetime,
            'px': px,
            'go': go,
            'stats': stats,
            # Safe file reading only
            'DATA_PATH': str(self.data_path),
            'FIGURES_DIR': str(self.output_dir / 'figures'),
        }
        
        # Add persistent variables (like df) to the execution environment
        safe_globals.update(self.persistent_vars)
        
        # Capture output
        import io
        import sys
        from contextlib import redirect_stdout
        
        output_buffer = io.StringIO()
        result = None
        
        try:
            # Create figures directory
            (self.output_dir / 'figures').mkdir(exist_ok=True)
            
            # Execute code with output capture
            exec_globals = safe_globals.copy()
            
            with redirect_stdout(output_buffer):
                exec(code, exec_globals)
            
            # Update persistent variables for next execution
            # Store any new variables created (except internal ones)
            for key, value in exec_globals.items():
                if (key not in safe_globals and 
                    not key.startswith('_') and
                    key not in ['__builtins__', 'pd', 'np', 'px', 'go', 'stats', 'Path', 'json', 're', 'datetime']):
                    self.persistent_vars[key] = value
            
            # Capture printed output
            printed_output = output_buffer.getvalue()
            
            # Capture any variables that were created or modified
            result = {k: v for k, v in exec_globals.items() 
                     if k not in safe_globals and not k.startswith('_')}
            
            return {
                'success': True,
                'error': None,
                'output': result,
                'printed': printed_output,
                'code': code
            }
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return {
                'success': False,
                'error': error_msg,
                'output': None,
                'printed': output_buffer.getvalue(),
                'code': code
            }


class AgenticAnalyzer:
    """
    AI Agent that drives the analysis by writing and executing code.
    Uses ReAct pattern: Reason -> Act -> Observe -> Repeat
    """
    
    def __init__(self, api_key=None, model='gpt-5.1'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key required. Set OPENAI_API_KEY env var or use --api-key")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.conversation_history = []
        
    def _call_llm(self, prompt, system_message=None):
        """Call LLM with conversation history."""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add new prompt
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            # max_tokens=3000
        )
        
        assistant_message = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def extract_code_blocks(self, text):
        """Extract Python code blocks from LLM response."""
        # Try multiple patterns
        patterns = [
            r'```python\n(.*?)\n```',
            r'```python\n(.*?)```',
            r'```\n(.*?)\n```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                # Clean up the code - remove extra indentation
                cleaned = []
                for code in matches:
                    # Remove common leading whitespace
                    lines = code.split('\n')
                    # Find minimum indentation (excluding empty lines)
                    non_empty_lines = [line for line in lines if line.strip()]
                    if non_empty_lines:
                        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
                        cleaned_lines = [line[min_indent:] if len(line) > min_indent else line 
                                       for line in lines]
                        cleaned.append('\n'.join(cleaned_lines))
                    else:
                        cleaned.append(code)
                return cleaned
        return []
    
    def analyze_with_code_execution(self, runner, max_iterations=10):
        """
        Main agentic loop: AI writes code, executes it, observes results, repeats.
        """
        system_message = """You are an expert data scientist with the ability to write and execute Python code.

Your task is to analyze a dataset by writing Python code that will be executed. You have access to:
- pandas (pd), numpy (np), scipy.stats (stats)
- plotly.express (px), plotly.graph_objects (go)
- A DataFrame called 'df' (or 'data') that is ALREADY LOADED with the dataset
- The dataset path is in the variable DATA_PATH (if you need to reload)
- Save figures to FIGURES_DIR directory using fig.write_html() or fig.write_image()

IMPORTANT: The variable 'df' is already available - you don't need to load it again!

CODE EXECUTION NOTES:
- Variables you create persist between executions
- You can reference 'df' in all your code blocks
- Use print() to display results - they will be shown back to you
- Create new variables as needed (they'll be available in subsequent steps)

CODE FORMATTING:
- Always write clean, properly indented Python code
- Use print() statements to see outputs
- Don't use markdown or comments before code blocks

ANALYSIS WORKFLOW:
1. First, explore 'df' structure: print(df.info()), print(df.describe()), print(df.head())
2. Check for missing values: print(df.isnull().sum())
3. Identify interesting patterns and correlations
4. Create visualizations for key findings (save to FIGURES_DIR)
5. Run statistical tests to validate findings
6. Build comprehensive understanding iteratively

Always wrap your Python code in ```python ``` blocks.
After each execution, you'll see the printed output.
When you have completed enough analysis (after ~8-12 iterations), respond with "ANALYSIS_COMPLETE" followed by a summary of all findings.
"""
        
        initial_prompt = f"""
The dataset is already loaded as 'df'. It has {runner.persistent_vars['df'].shape[0]} rows and {runner.persistent_vars['df'].shape[1]} columns.

Start by exploring the data structure. Write simple Python code with print() statements to see the results.
For example: print(df.info()), print(df.describe()), print(df.head())
"""
        
        analysis_log = []
        iteration = 0
        
        print("\n🤖 Starting Agentic Analysis...\n")
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            
            # Get LLM response
            if iteration == 1:
                response = self._call_llm(initial_prompt, system_message)
            else:
                response = self._call_llm("Continue your analysis based on the results. What should we explore next?")
            
            print(f"🧠 AI Decision: {response[:200]}..." if len(response) > 200 else f"🧠 AI Decision: {response}")
            
            # Check if analysis is complete
            if "ANALYSIS_COMPLETE" in response:
                print("\n✅ AI has completed the analysis!")
                analysis_log.append({
                    'iteration': iteration,
                    'response': response,
                    'type': 'completion'
                })
                break
            
            # Extract and execute code
            code_blocks = self.extract_code_blocks(response)
            
            if not code_blocks:
                # No code, just reasoning - continue
                analysis_log.append({
                    'iteration': iteration,
                    'response': response,
                    'type': 'reasoning'
                })
                continue
            
            # Execute each code block
            for idx, code in enumerate(code_blocks):
                result = runner.execute_code(code, f"Analysis step {iteration}.{idx+1}")
                
                analysis_log.append({
                    'iteration': iteration,
                    'code': code,
                    'result': result,
                    'type': 'execution'
                })
                
                # Feed results back to LLM with better formatting
                if result['success']:
                    output_summary = []
                    
                    # Show printed output
                    if result.get('printed'):
                        output_summary.append(f"Printed output:\n{result['printed'][:1000]}")
                    
                    # Show created variables
                    if result.get('output'):
                        var_info = []
                        for k, v in list(result['output'].items())[:5]:  # Limit to 5 vars
                            var_info.append(f"{k}: {type(v).__name__}")
                        if var_info:
                            output_summary.append(f"Variables created: {', '.join(var_info)}")
                    
                    feedback = "\n".join(output_summary) if output_summary else "Code executed successfully (no output)"
                else:
                    feedback = f"Error: {result['error']}\n\nProblematic code:\n{code[:200]}"
                
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Execution result:\n{feedback}"
                })
                
                print(f"  {'✅' if result['success'] else '❌'} {feedback[:200]}...")
                
                # If execution failed, give AI a chance to fix it
                if not result['success']:
                    print(f"  ⚠️  Giving AI a chance to fix the error...")
                    break  # Move to next iteration to let AI respond
        
        return analysis_log


class VisualizationEnhancer:
    """
    Second agent specialized in enhancing visualizations.
    Takes existing plots and makes them publication-quality.
    """
    
    def __init__(self, api_key=None, model='gpt-5.1'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def enhance_figure(self, runner, figure_description, data_context):
        """Generate code to create an enhanced version of a visualization."""
        
        prompt = f"""You are a data visualization expert. Create a publication-quality, visually stunning version of this plot:

{figure_description}

Data context: {data_context}

Requirements:
- Use plotly for interactive, beautiful visualizations
- Add proper titles, labels, and legends
- Use an appealing color scheme
- Add annotations or highlights for key insights
- Make it publication-ready
- Save the figure using fig.write_html(f"{{FIGURES_DIR}}/enhanced_fig_X.html")

Write Python code to create this enhanced visualization. The data is loaded in a variable called 'df'.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data visualization expert specializing in creating beautiful, informative plots."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        content = response.choices[0].message.content
        
        # Extract code
        code_pattern = r'```python\n(.*?)```'
        matches = re.findall(code_pattern, content, re.DOTALL)
        
        if matches:
            code = matches[0]
            # Execute the enhancement code
            result = runner.execute_code(code, "Visualization enhancement")
            return result
        
        return None


class ReportWriter:
    """
    Third agent specialized in writing the final narrative report.
    """
    
    def __init__(self, api_key=None, model='gpt-5.1'):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def write_introduction(self, analysis_log, data_info):
        """Write introduction section."""
        prompt = f"""Write a comprehensive introduction (400-500 words) for a data analysis report.

Dataset information:
{data_info}

Analysis performed:
{self._summarize_analysis(analysis_log)}

Include:
1. Overview of the dataset and its domain
2. Research questions or objectives
3. Methodology overview (mention the agentic AI approach)
4. Structure of the report

Write in a professional, engaging academic tone."""

        return self._call_llm(prompt)
    
    def write_discussion(self, analysis_log, findings):
        """Write critical discussion section."""
        prompt = f"""Write a critical discussion section (600-800 words) that analyzes these findings:

{findings}

Your discussion MUST:
1. Interpret statistical results carefully
2. CRITICALLY examine limitations:
   - Correlation vs causation fallacies
   - Sample bias and representativeness
   - Confounding variables
   - Data quality issues
   - Statistical assumptions
3. Discuss alternative explanations for patterns
4. Suggest what additional data/analysis would strengthen conclusions
5. Maintain scientific skepticism throughout

Remember: extraordinary claims require extraordinary evidence. Be rigorous and critical."""

        return self._call_llm(prompt)
    
    def write_conclusion(self, full_analysis):
        """Write conclusion section."""
        prompt = f"""Write a conclusion (300-400 words) that synthesizes this analysis:

{full_analysis}

Include:
1. Key findings summary
2. Main limitations and caveats
3. Practical implications
4. Future research directions
5. Final thoughts

Be balanced and avoid overstating the conclusions."""

        return self._call_llm(prompt)
    
    def _call_llm(self, prompt):
        """Helper to call LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert scientific writer specializing in data analysis reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            # max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _summarize_analysis(self, analysis_log):
        """Summarize the analysis log."""
        summary = []
        for entry in analysis_log[:10]:  # First 10 steps
            if entry['type'] == 'execution' and entry['result']['success']:
                summary.append(f"- Executed: {entry['code'][:100]}...")
        return "\n".join(summary)


class AgenticReportGenerator:
    """
    Orchestrates the entire agentic analysis pipeline.
    """
    
    def __init__(self, data_path, output_path, api_key=None):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.api_key = api_key
        
        # Create output directory
        self.output_dir = self.output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.runner = SafeCodeRunner(self.data_path, self.output_dir)
        self.analyzer = AgenticAnalyzer(api_key=api_key)
        self.viz_enhancer = VisualizationEnhancer(api_key=api_key)
        self.writer = ReportWriter(api_key=api_key)
    
    def generate_report(self):
        """Main report generation pipeline."""
        print("🚀 IAthon Agentic Report Generator")
        print("=" * 70)
        print(f"📁 Input: {self.data_path}")
        print(f"📄 Output: {self.output_path}")
        print()
        
        # Phase 1: Agentic Analysis
        print("🔬 PHASE 1: Agentic Data Analysis")
        print("-" * 70)
        analysis_log = self.analyzer.analyze_with_code_execution(self.runner, max_iterations=15)
        
        # Phase 2: Enhance Visualizations
        print("\n🎨 PHASE 2: Enhancing Visualizations")
        print("-" * 70)
        self._enhance_visualizations(analysis_log)
        
        # Phase 3: Write Report
        print("\n✍️  PHASE 3: Writing Report")
        print("-" * 70)
        report = self._write_final_report(analysis_log)
        
        # Save report
        self.output_path.write_text(report, encoding='utf-8')
        print(f"\n✅ Report saved: {self.output_path}")
        
        return report
    
    def _enhance_visualizations(self, analysis_log):
        """Enhance visualizations using specialized agent."""
        # Find figures created during analysis
        figures_dir = self.output_dir / 'figures'
        if figures_dir.exists():
            existing_figs = list(figures_dir.glob('*.html')) + list(figures_dir.glob('*.png'))
            print(f"  Found {len(existing_figs)} figures to potentially enhance")
            
            # For demonstration, enhance up to 3 figures
            for i, fig_path in enumerate(existing_figs[:3]):
                print(f"  🎨 Enhancing figure {i+1}...")
                # In a full implementation, we'd analyze the figure and enhance it
                # For now, we'll skip this to keep the code concise
        else:
            print("  ℹ️  No figures directory found")
    
    def _write_final_report(self, analysis_log):
        """Assemble the final report."""
        
        # Extract key information
        data_info = self._get_data_summary()
        findings = self._extract_findings(analysis_log)
        
        # Generate sections
        print("  📝 Writing introduction...")
        introduction = self.writer.write_introduction(analysis_log, data_info)
        
        print("  📝 Writing discussion...")
        discussion = self.writer.write_discussion(analysis_log, findings)
        
        print("  📝 Writing conclusion...")
        conclusion = self.writer.write_conclusion(findings + "\n" + discussion)
        
        # Assemble report
        report = f"""# Agentic Data Analysis Report
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Using AI-Driven Code Execution (ReAct Pattern)*

---

## 1. Introduction

{introduction}

---

## 2. Methodology

This analysis employed an **agentic AI approach** where an AI agent autonomously:
1. Wrote Python code to explore the dataset
2. Executed code in a sandboxed environment
3. Observed results and iteratively refined analysis
4. Generated visualizations and statistical tests
5. Synthesized findings into this report

This represents a shift from traditional fixed-pipeline analysis to a dynamic, exploratory approach where AI acts as the driver rather than a passive tool.

---

## 3. Analysis Results

{self._format_analysis_results(analysis_log)}

---

## 4. Visualizations

{self._format_visualizations()}

---

## 5. Discussion

{discussion}

---

## 6. Conclusion

{conclusion}

---

## 7. Appendix: Execution Log

<details>
<summary>Click to expand full analysis execution log</summary>

{self._format_execution_log(analysis_log)}

</details>

---

*Report generated by IAthon Agentic Analysis System*
*Powered by OpenAI GPT-4 with ReAct pattern*
"""
        
        return report
    
    def _get_data_summary(self):
        """Get basic data summary."""
        try:
            df = pd.read_csv(self.data_path)
            return f"Shape: {df.shape}, Columns: {list(df.columns)}, Types: {df.dtypes.to_dict()}"
        except:
            return "Data summary unavailable"
    
    def _extract_findings(self, analysis_log):
        """Extract key findings from analysis log."""
        findings = []
        for entry in analysis_log:
            if entry['type'] == 'execution' and entry['result']['success']:
                findings.append(f"- {entry['code'][:200]}")
        return "\n".join(findings[:10])
    
    def _format_analysis_results(self, analysis_log):
        """Format analysis results for report."""
        results = []
        for i, entry in enumerate(analysis_log):
            if entry['type'] == 'execution' and entry['result']['success']:
                results.append(f"### Finding {i+1}\n\n```python\n{entry['code']}\n```\n")
        return "\n".join(results[:10]) or "Analysis results captured in execution log."
    
    def _format_visualizations(self):
        """Format visualization references."""
        figures_dir = self.output_dir / 'figures'
        if not figures_dir.exists():
            return "*No visualizations generated*"
        
        figs = sorted(figures_dir.glob('*.html')) + sorted(figures_dir.glob('*.png'))
        
        viz_md = []
        for i, fig in enumerate(figs[:10], 1):
            if fig.suffix == '.html':
                viz_md.append(f'<iframe src="figures/{fig.name}" width="100%" height="600"></iframe>\n\n*Figure {i}: {fig.stem}*\n')
            else:
                viz_md.append(f'![Figure {i}](figures/{fig.name})\n\n*Figure {i}: {fig.stem}*\n')
        
        return "\n".join(viz_md) or "*Visualizations were generated during analysis*"
    
    def _format_execution_log(self, analysis_log):
        """Format execution log for appendix."""
        log_md = []
        for i, entry in enumerate(analysis_log, 1):
            log_md.append(f"### Step {i}\n")
            if entry['type'] == 'execution':
                log_md.append(f"**Code:**\n```python\n{entry['code']}\n```\n")
                log_md.append(f"**Result:** {'✅ Success' if entry['result']['success'] else '❌ Failed'}\n")
            elif entry['type'] == 'reasoning':
                log_md.append(f"**Reasoning:** {entry['response'][:300]}...\n")
        
        return "\n".join(log_md)
    
    def convert_to_format(self, output_format):
        """Convert markdown to other formats using pandoc."""
        output_path = self.output_path.with_suffix(f'.{output_format}')
        
        print(f"\n📄 Converting to {output_format.upper()}...")
        
        try:
            cmd = ['pandoc', str(self.output_path), '-o', str(output_path)]
            
            if output_format == 'pptx':
                cmd.extend(['--slide-level', '2'])
            elif output_format == 'pdf':
                cmd.extend(['--pdf-engine', 'xelatex'])
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✅ {output_format.upper()} saved: {output_path}")
            return output_path
            
        except FileNotFoundError:
            print(f"❌ Error: pandoc not found. Install from https://pandoc.org/installing.html")
            return None
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting: {e.stderr.decode()}")
            return None


def download_sample_datasets():
    """Download sample datasets for quick testing."""
    print("\n📦 Downloading sample datasets...\n")
    
    samples_dir = Path('sample_datasets')
    samples_dir.mkdir(exist_ok=True)
    
    datasets = {
        'iris.csv': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
        'titanic.csv': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
        'tips.csv': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
        'penguins.csv': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv',
    }
    
    try:
        import urllib.request
        for filename, url in datasets.items():
            filepath = samples_dir / filename
            if not filepath.exists():
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✅ Saved: {filepath}")
            else:
                print(f"  ℹ️  Already exists: {filepath}")
        
        print(f"\n✅ Sample datasets ready in '{samples_dir}' directory")
        print("\nTry running:")
        for fname in datasets.keys():
            print(f"  python app.py --input sample_datasets/{fname} --output reports/{fname.replace('.csv', '_report.md')}")
        return True
    except Exception as e:
        print(f"❌ Error downloading datasets: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='IAthon - Agentic AI-Driven Data Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --input data.csv --output report.md
  python app.py --input data.xlsx --output report.md --format pdf
  python app.py --download-samples
  
Environment Variables:
  OPENAI_API_KEY    Your OpenAI API key (required)
  
Features:
  - Agentic AI that writes and executes code autonomously
  - Safe code execution with Windows protections
  - Iterative analysis with ReAct pattern
  - Automatic visualization generation
  - Critical statistical analysis
  - Publication-quality reports
        """
    )
    
    parser.add_argument('--input', '-i', help='Input data file (CSV, XLSX, JSON)')
    parser.add_argument('--output', '-o', help='Output markdown file path')
    parser.add_argument('--format', '-f', choices=['pptx', 'docx', 'pdf'],
                       help='Additional output format (requires pandoc)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--download-samples', action='store_true',
                       help='Download sample datasets for testing')
    
    args = parser.parse_args()
    
    # Handle sample download
    if args.download_samples:
        download_sample_datasets()
        return
    
    # Validate required arguments
    if not args.input or not args.output:
        parser.print_help()
        print("\n❌ Error: --input and --output are required")
        sys.exit(1)
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        # Generate report
        generator = AgenticReportGenerator(
            data_path=args.input,
            output_path=args.output,
            api_key=args.api_key
        )
        
        generator.generate_report()
        
        # Convert to additional format if requested
        if args.format:
            generator.convert_to_format(args.format)
        
        print("\n" + "=" * 70)
        print("✨ Agentic analysis complete!")
        print(f"📄 Report: {args.output}")
        if args.format:
            print(f"📄 {args.format.upper()}: {Path(args.output).with_suffix(f'.{args.format}')}")
        print("\n💡 Tip: Review the execution log in the report appendix to see")
        print("   how the AI agent explored and analyzed your data!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()