import argparse
import optuna
import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def objective(trial):
    hyp = {
        "lr0": trial.suggest_float("lr0", 1e-3, 1e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.80, 0.97),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "degrees": 15.0,
        "translate": 0.2,
        "scale": 0.75,
        "shear": 0.0,
        "mosaic": 1.0,
        "mixup": 0.4,
        "auto_augment": "simple",
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }


    metrics = YOLO("yolo11m.pt").train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=0.45,
        device=DEVICE,
        **hyp,
        cos_lr=True,    
        lrf=0.2,        
        augment=True,
        rect=False,
        plots=False
    )

    torch.cuda.empty_cache()
    return metrics.box.map

def tune_hyperparameters(n_trials=10):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparameters:", study.best_params)
    return study.best_params

def final_training(best_hyp):
    YOLO("best.pt").train(
        data="data.yaml",
        epochs=1,
        imgsz=640,
        batch=0.5,
        device=DEVICE,
        optimizer="AdamW",
        cos_lr=True,
        lrf=0.2,
        **best_hyp,
        augment=True,
        rect=False,
        plots=True,
        time=12,
        val=True
    )

def main():
    parser = argparse.ArgumentParser("YOLO Aug")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning on YOLO")
    args = parser.parse_args()

    if args.tune:
        best = tune_hyperparameters(n_trials=10)
    else:
        best = {
            "lr0": 0.011348857912055617,
            "momentum": 0.8679822426370379,
            "weight_decay": 3.251856394748595e-06,
            "degrees": 15.0,
            "translate": 0.2,
            "scale": 0.75,
            "shear": 0.0,
            "mosaic": 1.0,
            "mixup": 0.4,
            "auto_augment": "simple",
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        }
        print("Running final training with:", best)
        final_training(best)

if __name__ == "__main__":
    main()
