"""Shared plumbing: the frozen CV scheme, column lists, paths.

THE FROZEN FOLDS
----------------
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

This is deliberately identical to the scheme used by the public
`szymonkapiski/s6e8-oof-library-47-models` dataset (74 models, original row order).
Matching it means our own out-of-fold predictions are directly stackable against that
library without leaking. Never change it: every saved oof_*.npy becomes invalid if the
fold assignment moves, and the stack silently becomes dishonest.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
SEED = 42
TARGET = "addicted_label"
ID = "id"

NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "notifications_per_day",
       "app_opens_per_day", "weekend_screen_time"]
CAT = ["gender", "stress_level", "academic_work_impact"]
# The generator identity: daily >= social + gaming + work_study, with zero violations.
COMP = ["social_media_hours", "gaming_hours", "work_study_hours"]
DAILY = "daily_screen_time_hours"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LIB = os.path.join(DATA, "oof")           # the public 74-model OOF library
OOF = os.path.join(ROOT, "oof")           # our own predictions
CACHE = os.path.join(ROOT, "cache")       # per-fold target-encoding caches
SUB = os.path.join(ROOT, "submissions")

for _d in (OOF, CACHE, SUB):
    os.makedirs(_d, exist_ok=True)


def load_raw():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    return tr, te


def get_folds(y):
    """The frozen fold assignment. Returns a list of (train_idx, valid_idx)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


def save_preds(name, oof, test, n_train, n_test, out=OOF):
    """Persist OOF + test predictions in ORIGINAL ROW ORDER as float64."""
    oof = np.asarray(oof, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    assert oof.shape == (n_train,), f"{name}: oof {oof.shape} != ({n_train},)"
    assert test.shape == (n_test,), f"{name}: test {test.shape} != ({n_test},)"
    assert np.isfinite(oof).all() and np.isfinite(test).all(), f"{name}: non-finite"
    np.save(os.path.join(out, f"oof_{name}.npy"), oof)
    np.save(os.path.join(out, f"test_{name}.npy"), test)
    print(f"[saved] {name} -> {out}", flush=True)
