#!/bin/sh
pip install --upgrade pip
pip install -r requirements.txt
python runai_hpo.py $@
