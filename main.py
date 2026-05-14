import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR / "src"))

from weather_ml_project.pipeline import run_pipeline


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
