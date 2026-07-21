from src.pipeline.training_pipeline import run_training_pipeline

DEFAULT_STOCK = "PETR4.SA"


def main():
    run_training_pipeline(DEFAULT_STOCK)


if __name__ == "__main__":
    main()