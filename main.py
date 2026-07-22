import argparse

from src.pipeline.training_pipeline import run_training_pipeline


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stock",
        default="PETR4.SA",
        help="Ticker da ação"
    )

    args = parser.parse_args()

    run_training_pipeline(args.stock)


if __name__ == "__main__":
    main()