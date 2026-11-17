from src.data_loader import load_data
from src.experiment import run_experiment


if __name__ == "__main__":
    data, drift_at, synthetic_mode = load_data()
    print("Rows:", len(data))
    print("\nClass distribution:\n", data["Label"].value_counts().to_string())
    run_experiment(data, drift_at, synthetic_mode)
