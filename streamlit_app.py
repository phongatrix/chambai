# -*- coding: utf-8 -*-
"""
Entrypoint for Streamlit Cloud (https://fengcheck.streamlit.app).
Chạy webapp.py chính của hệ thống OMR.
"""
import runpy
from pathlib import Path

webapp_file = Path(__file__).resolve().parent / "webapp.py"
runpy.run_path(str(webapp_file), run_name="__main__")
