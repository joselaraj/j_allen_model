#!/bin/bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install nfl_data_py --no-deps
pip install -r requirements.txt