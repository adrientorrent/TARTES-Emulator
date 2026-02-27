#!/usr/bin/env python3

import optuna

study_path = "/home/torrenta/new-TARTES-Emulator/finetuning/study.db"
study = optuna.create_study(
    direction="minimize",
    study_name="TARTES",
    storage=f"sqlite:///{study_path}",
    load_if_exists=True
)
print("Id:", study.best_trial._trial_id)
print("Hyperpameters:", study.best_params)
print("MSE:", study.best_value)
