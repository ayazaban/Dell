import sys
import warnings
from pathlib import Path


# Suppress noisy numpy RuntimeWarnings from aggregation of sparse terrain data
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from weather_ml_project.pipeline import run_pipeline


def main() -> None:
    print("Demarrage du pipeline...", flush=True)
    run_pipeline()


if __name__ == "__main__":
    main()
