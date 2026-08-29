#!/bin/bash
cd /home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
exec .venv/bin/python experiments/w93a_suite.py > experiments/w113_suite2.log 2>&1
