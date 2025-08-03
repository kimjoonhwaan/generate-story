#!/usr/bin/env python3
"""
RAG Story Generator - Main Entry Point

This script provides both CLI and web interface options for the RAG story generation system.
"""

import sys
import argparse
from rag_system import RAGSystem

def run_cli():
    """Run the command-line interface"""
    print("Starting RAG Story Generator CLI...")
    
    # Initialize system
    try:
        rag_system = RAGSystem(use_openai=True)
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        print("Falling back to local generation...")
        rag_system = RAGSystem(use_openai=False)
    
    # Run interactive mode
    rag_system.interactive_story_generation()

def run_web():
    """Run the web interface using Streamlit"""
    import subprocess
    import os
    
    print("Starting RAG Story Generator Web Interface...")
    print("This will open in your default web browser.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start web interface: {e}")
        print("Make sure streamlit is installed: pip install streamlit")
    except KeyboardInterrupt:
        print("\nWeb interface stopped.")

def main():
    parser = argparse.ArgumentParser(
        description="RAG Story Generator - Create stories using keywords and document context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run web interface (default)
  python main.py --web             # Run web interface
  python main.py --cli             # Run command-line interface
  
For the web interface, navigate to http://localhost:8501 in your browser.
        """
    )
    
    parser.add_argument(
        "--cli", 
        action="store_true", 
        help="Run the command-line interface instead of web interface"
    )
    
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Run the web interface (default)"
    )
    
    args = parser.parse_args()
    
    # Default to web interface if no arguments provided
    if not args.cli and not args.web:
        args.web = True
    
    if args.cli:
        run_cli()
    elif args.web:
        run_web()

if __name__ == "__main__":
    main() 