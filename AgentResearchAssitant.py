#!/usr/bin/env python3
"""
IAthon – Iteration 6.6 (Token Tracking Version - Fixed)
New: Comprehensive token usage logging and cost estimation.
"""

import argparse
import os
import shutil      
import subprocess
import re
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict

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

# Try to import python-pptx for template creation
try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False

# ======================================================
# Token Usage Tracker
# ======================================================
@dataclass
class TokenUsage:
    """Track token usage per model"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += (prompt + completion)

@dataclass
class UsageTracker:
    """Central tracker for all API usage"""
    models: Dict[str, TokenUsage] = field(default_factory=dict)
    pricing_cache: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def fetch_pricing(self):
        """Fetch current pricing from OpenAI (with fallback to hardcoded values)"""
        # Fallback pricing (updated from your link - Jan 2025)
        fallback_pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},  # per token
            "gpt-4o-mini": {"input": 0.150 / 1_000_000, "output": 0.600 / 1_000_000},
            "o1": {"input": 15.00 / 1_000_000, "output": 60.00 / 1_000_000},
            "o1-mini": {"input": 3.00 / 1_000_000, "output": 12.00 / 1_000_000},
            "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
            "gpt-4": {"input": 30.00 / 1_000_000, "output": 60.00 / 1_000_000},
            "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
        }
        
        # Try to map model names to pricing (handle version suffixes)
        self.pricing_cache = {}
        for model_name in self.models.keys():
            # Extract base model name (e.g., "gpt-4o" from "gpt-4o-2024-11-20")
            base_model = None
            for known_model in fallback_pricing.keys():
                if model_name.startswith(known_model):
                    base_model = known_model
                    break
            
            if base_model:
                self.pricing_cache[model_name] = fallback_pricing[base_model]
            else:
                # Unknown model - use conservative estimate
                print(f"⚠️  Unknown pricing for model '{model_name}', using GPT-4 rates as estimate")
                self.pricing_cache[model_name] = fallback_pricing["gpt-4"]
        
        return self.pricing_cache
    
    def record(self, model: str, usage):
        """Record usage from OpenAI response - handles multiple API formats"""
        if model not in self.models:
            self.models[model] = TokenUsage()
        
        # Try different attribute names for different API endpoints
        prompt = 0
        completion = 0
        
        # Try to get tokens using various possible attribute names
        for prompt_attr in ['prompt_tokens', 'input_tokens', 'total_input_tokens']:
            if hasattr(usage, prompt_attr):
                prompt = getattr(usage, prompt_attr)
                break
        
        for completion_attr in ['completion_tokens', 'output_tokens', 'total_output_tokens']:
            if hasattr(usage, completion_attr):
                completion = getattr(usage, completion_attr)
                break
        
        # Fallback: check if it's a dict-like object
        if prompt == 0 and completion == 0:
            try:
                if hasattr(usage, '__dict__'):
                    usage_dict = usage.__dict__
                elif hasattr(usage, 'model_dump'):
                    usage_dict = usage.model_dump()
                else:
                    usage_dict = dict(usage)
                
                prompt = usage_dict.get('prompt_tokens') or usage_dict.get('input_tokens') or 0
                completion = usage_dict.get('completion_tokens') or usage_dict.get('output_tokens') or 0
            except:
                print(f"⚠️  Warning: Could not extract token usage from response")
                print(f"   Usage object type: {type(usage)}")
                print(f"   Available attributes: {dir(usage)}")
        
        self.models[model].add(prompt, completion)
    
    def print_summary(self):
        """Print detailed usage summary with cost estimates"""
        print("\n" + "="*70)
        print("📊 TOKEN USAGE SUMMARY")
        print("="*70)
        
        # Fetch current pricing
        pricing = self.fetch_pricing()
        
        total_cost = 0.0
        
        for model, usage in self.models.items():
            print(f"\n🤖 Model: {model}")
            print(f"   Prompt tokens:     {usage.prompt_tokens:,}")
            print(f"   Completion tokens: {usage.completion_tokens:,}")
            print(f"   Total tokens:      {usage.total_tokens:,}")
            
            # Cost calculation
            if model in pricing:
                input_cost = usage.prompt_tokens * pricing[model]["input"]
                output_cost = usage.completion_tokens * pricing[model]["output"]
                model_cost = input_cost + output_cost
                total_cost += model_cost
                
                print(f"   Input cost:        ${input_cost:.6f}")
                print(f"   Output cost:       ${output_cost:.6f}")
                print(f"   Total cost:        ${model_cost:.6f}")
            else:
                print(f"   Cost:              Unknown pricing for this model")
        
        print(f"\n{'='*70}")
        print(f"💰 TOTAL ESTIMATED COST: ${total_cost:.6f}")
        print(f"   (Based on OpenAI pricing as of Jan 2025)")
        print("="*70)
        
        return total_cost
    
    def save_log(self, output_dir: Path):
        """Save detailed log to file"""
        log_path = output_dir / "token_usage.log"
        
        pricing = self.pricing_cache if self.pricing_cache else self.fetch_pricing()
        
        with open(log_path, 'w') as f:
            f.write("TOKEN USAGE LOG\n")
            f.write("="*70 + "\n\n")
            
            total_cost = 0.0
            for model, usage in self.models.items():
                f.write(f"Model: {model}\n")
                f.write(f"  Prompt tokens:     {usage.prompt_tokens:,}\n")
                f.write(f"  Completion tokens: {usage.completion_tokens:,}\n")
                f.write(f"  Total tokens:      {usage.total_tokens:,}\n")
                
                if model in pricing:
                    input_cost = usage.prompt_tokens * pricing[model]["input"]
                    output_cost = usage.completion_tokens * pricing[model]["output"]
                    model_cost = input_cost + output_cost
                    total_cost += model_cost
                    
                    f.write(f"  Input cost:        ${input_cost:.6f}\n")
                    f.write(f"  Output cost:       ${output_cost:.6f}\n")
                    f.write(f"  Total cost:        ${model_cost:.6f}\n")
                
                f.write("\n")
            
            f.write(f"{'='*70}\n")
            f.write(f"TOTAL COST: ${total_cost:.6f}\n")
        
        print(f"📝 Token usage log saved to: {log_path}")

# ======================================================
# Utilities
# ======================================================
def get_api_key(cli_key=None):
    return cli_key or os.getenv("OPENAI_API_KEY")

def robust_rmtree(path: Path):
    """Attempt to delete a directory, handling Windows/OneDrive locks."""
    if not path.exists():
        return
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1)
    for item in path.iterdir():
        try:
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)
        except:
            pass

def create_styled_reference(output_dir: Path, figures_dir: Path):
    """Create a styled PowerPoint reference template"""
    if not PPTX_AVAILABLE:
        print("⚠️  python-pptx not available. Install with: pip install python-pptx")
        print("   Using default PowerPoint styling instead.")
        return None
    
    print("🎨 Creating styled presentation template...")
    
    # Start with a blank presentation
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Define color scheme (professional blue theme)
    DARK_BLUE = RGBColor(31, 78, 121)
    ACCENT_BLUE = RGBColor(91, 155, 213)
    GRAY = RGBColor(89, 89, 89)
    
    # Add one sample slide of each type Pandoc expects
    # Slide 1: Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide1 = prs.slides.add_slide(title_slide_layout)
    
    # Customize title slide text
    title = slide1.shapes.title
    if title:
        title.text = "Sample Title"
        title.text_frame.paragraphs[0].font.size = Pt(44)
        title.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE
        title.text_frame.paragraphs[0].font.bold = True
    
    # Slide 2: Title and Content
    content_slide_layout = prs.slide_layouts[1]
    slide2 = prs.slides.add_slide(content_slide_layout)
    
    # Customize content slide
    title2 = slide2.shapes.title
    if title2:
        title2.text = "Sample Slide"
        title2.text_frame.paragraphs[0].font.size = Pt(32)
        title2.text_frame.paragraphs[0].font.color.rgb = DARK_BLUE
        title2.text_frame.paragraphs[0].font.bold = True
    
    # Find and style the content placeholder
    for shape in slide2.placeholders:
        if shape.placeholder_format.type == 2:  # Body placeholder
            tf = shape.text_frame
            tf.text = "Sample bullet point"
            for paragraph in tf.paragraphs:
                paragraph.font.size = Pt(18)
                paragraph.font.color.rgb = GRAY
    
    # Save reference template
    ref_path = output_dir / "reference_template.pptx"
    prs.save(str(ref_path))
    print(f"✅ Template created: {ref_path}")
    return ref_path

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
    def __init__(self, api_key, tracker: UsageTracker, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.tracker = tracker
        self.verbose = verbose
        self.model = "gpt-5.1"
    
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
            model=self.model, 
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.7 
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            self.tracker.record(self.model, response.usage)
        
        return response.choices[0].message.content
    
    def synthesize_presentation(self, analysis_log, figures_dir, data_preview):
        """Generate presentation-optimized content with bullet points"""
        if self.verbose: print("📊 Creating presentation slides...")
        
        figs = sorted(figures_dir.glob("*.png"))
        fig_list = "\n".join([f"- {f.name}" for f in figs])

        pptx_prompt = f"""
        You are creating a professional PowerPoint presentation for executives.
        
        DATA PREVIEW:
        {data_preview}

        ANALYSIS HISTORY:
        {analysis_log}

        AVAILABLE FIGURES (CRITICAL - You MUST use these exact filenames):
        {fig_list}

        YOUR TASK - Create a slide deck in Markdown format:
        
        FIRST: Analyze the data context and identify the domain (e.g., sports, finance, healthcare, etc.)
        
        STRUCTURE (Use # for slide titles):
        
        # [Title Slide - Catchy title based on data context]
        
        [Subtitle with context]
        
        **Domain Context Image**: [Describe what kind of image would represent this domain - be specific, e.g., "triathlon athlete crossing finish line", "stock market trading floor", "hospital emergency room"]
        
        ---
        
        # Executive Summary
        
        - 3-4 key bullet points (high-level insights only)
        - Each bullet should be ONE concise line
        
        ---
        
        # Data Overview
        
        - What this data represents
        - Key metrics tracked
        - Sample size and scope
        
        ---
        
        # Key Finding 1: [Descriptive Title]
        
        - 2-3 bullet points summarizing the insight
        - Keep each bullet to ONE line
        
        ![](figures/EXACT_FIGURE_NAME_1.png)
        
        ---
        
        # Key Finding 2: [Descriptive Title]
        
        - 2-3 bullet points
        
        ![](figures/EXACT_FIGURE_NAME_2.png)
        
        ---
        
        [Continue for EACH figure in the list above - create one findings slide per figure]
        
        # Statistical Insights
        
        - 3-4 bullets highlighting statistical significance
        - Correlations or patterns discovered
        
        ---
        
        # Conclusions & Recommendations
        
        - 3-4 actionable takeaways
        - Each as a single, impactful bullet
        
        CRITICAL RULES FOR IMAGES:
        - Use EXACTLY: ![](figures/filename.png) - NO alt text in brackets
        - Use the EXACT filenames from the list above
        - Place image AFTER the bullet points on each findings slide
        - Create ONE slide per figure
        - Use --- to separate slides
        - Include the domain context image description for the title slide
        
        OTHER RULES:
        - Maximum 5 bullets per slide
        - Each bullet must be ONE line (max 15 words)
        - Use domain-appropriate terminology
        - Be specific with numbers when relevant
        - NO paragraphs, NO long explanations
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": pptx_prompt}],
            temperature=0.6
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            self.tracker.record(self.model, response.usage)
        
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
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        # Track usage
        if hasattr(response, 'usage'):
            self.tracker.record(self.model, response.usage)
        
        return response.choices[0].message.content.strip()

# ======================================================
# 4. CODE GENERATOR (CODEX)
# ======================================================
class CodexGenerator:
    def __init__(self, api_key, tracker: UsageTracker, verbose=False):
        self.client = OpenAI(api_key=api_key)
        self.tracker = tracker
        self.verbose = verbose
        self.model = "gpt-5.1-codex-max"

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
            model=self.model,
            input=prompt,
        )
        
        # Track usage - try to access usage attribute safely
        if hasattr(response, 'usage') and response.usage is not None:
            self.tracker.record(self.model, response.usage)
        
        return response.output_text.strip()

# ======================================================
# 5. REACT ORCHESTRATOR
# ======================================================
class ReActAnalyzer:
    def __init__(self, runner, decider, coder, tracker: UsageTracker, max_steps=20, verbose=False):
        self.runner = runner
        self.decider = decider
        self.coder = coder
        self.tracker = tracker
        self.max_steps = max_steps
        self.verbose = verbose
        self.observations = []
        self.analysis_log = []
        self.data_preview = self.runner.df.head(10).to_string()

    def observe(self):
        figs = sorted(self.runner.figures_dir.glob("*.png"))
        obs = f"- Figures: {len(figs)}\n- Files: " + ", ".join([f.name for f in figs]) if figs else "- No figs."
        self.observations.append(obs)

    def run(self, user_requirements): 
        print(f"\n🔬 STARTING ANALYSIS (Iteration 6.6 - Token Tracking)\n{'='*70}")
        self.observe()
        
        for step in range(self.max_steps):
            print(f"\n🔄 STEP {step + 1}/{self.max_steps}")
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
        
        return report_path, report_text

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

    # Initialize usage tracker
    tracker = UsageTracker()

    df, num_features, cat_features = load_and_clean(input_path)
    runner = SafeRunner(df, num_features, cat_features, output_dir, verbose=args.verbose)
    decider = DecisionMaker(api_key, tracker, verbose=args.verbose)
    coder = CodexGenerator(api_key, tracker, verbose=args.verbose)

    analyzer = ReActAnalyzer(runner, decider, coder, tracker, max_steps=5, verbose=args.verbose)
    report_md_path, report_text = analyzer.run("Comprehensive domain-specific report with embedded figures.") 

    if args.format:
        print(f"📦 Converting to {args.format}...")
        out_file = output_path.with_suffix(f".{args.format}")
        
        if args.format == "pptx":
            # Generate presentation-optimized content
            pptx_content = decider.synthesize_presentation(
                analyzer.analysis_log, 
                runner.figures_dir, 
                analyzer.data_preview
            )
            pptx_md_path = output_dir / "presentation.md"
            pptx_md_path.write_text(pptx_content, encoding='utf-8')
            print(f"📊 Presentation markdown saved to {pptx_md_path}")
            
            # Extract domain image suggestion from content
            domain_image_match = re.search(r'\*\*Domain Context Image\*\*:\s*(.+?)(?:\n|$)', pptx_content)
            if domain_image_match:
                domain_context = domain_image_match.group(1).strip()
                print(f"💡 Suggested title image: {domain_context}")
                print(f"   → Search for this on Unsplash/Pexels and add to title slide manually")
            
            # Create reference document with styling
            ref_path = create_styled_reference(output_dir, runner.figures_dir)
            
            # Build Pandoc command
            pandoc_cmd = ["pandoc"]
            
            # Input file (must be relative to cwd)
            pandoc_cmd.extend([str(pptx_md_path.name)])
            
            # Output format and file
            pandoc_cmd.extend(["-t", "pptx", "-o", str(out_file.name)])
            
            # Add reference doc if template was created successfully
            if ref_path and ref_path.exists():
                pandoc_cmd.extend([f"--reference-doc={ref_path.name}"])
            
            # Resource path for images (absolute path is safer)
            pandoc_cmd.extend([f"--resource-path=.:figures:{runner.figures_dir.absolute()}"])
            
            # Add standalone flag to ensure complete document
            pandoc_cmd.append("--standalone")
            
            print(f"🔧 Running: {' '.join(pandoc_cmd)}")
            
            try:
                result = subprocess.run(
                    pandoc_cmd, 
                    cwd=output_dir, 
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✅ Presentation saved to {out_file}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Pandoc error: {e}")
                print(f"   stdout: {e.stdout}")
                print(f"   stderr: {e.stderr}")
                print(f"   Trying without reference doc...")
                
                # Fallback: try without reference doc
                pandoc_cmd_simple = [
                    "pandoc",
                    str(pptx_md_path.name),
                    "-t", "pptx",
                    "-o", str(out_file.name),
                    f"--resource-path=.:figures:{runner.figures_dir.absolute()}",
                    "--standalone"
                ]
                subprocess.run(pandoc_cmd_simple, cwd=output_dir, check=True)
                print(f"✅ Presentation saved to {out_file} (with default styling)")
        else:
            # For PDF and DOCX, use the full report
            subprocess.run(
                ["pandoc", report_md_path.name, "-o", out_file.name],
                cwd=output_dir
            )
            print(f"✅ Document saved to {out_file}")
    
    # Print token usage LAST (after all processing is complete)
    print("\n" + "="*70)
    tracker.print_summary()
    tracker.save_log(output_dir)

if __name__ == "__main__":
    main()