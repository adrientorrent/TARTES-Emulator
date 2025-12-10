#!/usr/bin/env python3

import optuna

study_path = "/home/torrenta/TARTES-Emulator/scripts/finetuning/study.db"
study = optuna.create_study(
    direction="minimize",
    study_name="TARTES",
    storage=f"sqlite:///{study_path}",
    load_if_exists=True
)
print(study.best_params, study.best_trial._trial_id, study.best_value)
