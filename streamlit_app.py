# -*- coding: utf-8 -*-
"""
Entrypoint for Streamlit Cloud (https://fengcheck.streamlit.app).
Chạy webapp.py chính của hệ thống OMR.
"""
from pathlib import Path

webapp_file = Path(__file__).resolve().parent / "webapp.py"
with open(webapp_file, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(webapp_file), "exec")
    exec(code, globals())
