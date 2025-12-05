#!/usr/bin/env python3

import optuna

study = optuna.create_study(
    direction="minimize",
    study_name="TARTES",
    storage="sqlite:///TARTES.db",
    load_if_exists=True
)
print(study.best_params)
