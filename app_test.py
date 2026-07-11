import streamlit as st
import sys
import os

# Reconfigure stdout to use UTF-8 to avoid encoding issues in logs
sys.stdout.reconfigure(encoding='utf-8')

# Link to your existing src directory
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

from sentiment import _load_finbert

st.set_page_config(page_title="FinBERT Latency & Memory Test", layout="centered")

st.title("🧪 FinBERT Cloud Memory & Latency Test")
st.write("This app loads and runs the ProsusAI/finbert model locally inside the container to test RAM usage and latency.")

if st.button("Load and Run Model", type="primary"):
    with st.spinner("Attempting to load FinBERT into container RAM (first run downloads ~400MB)..."):
        try:
            # 1. Load the model using your existing local loader
            pipe = _load_finbert()
            
            if pipe is not None:
                st.success("✅ Model loaded successfully into RAM!")
                
                # 2. Run a test prediction
                result = pipe("Reliance profit jumps 150%, beating all estimates.")
                st.write("### Output Prediction:")
                st.json(result)
            else:
                st.error("❌ Model loader returned None.")
        except Exception as e:
            st.error(f"❌ Crash details: {e}")
