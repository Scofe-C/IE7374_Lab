import sys
import pandas as pd


def main():
    print("--- IE7374 Lab 1: Automated Data Fetching ---")

    # Download Iris dataset directly from a public URL
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

    try:
        print(f"Fetching data from: {url}")
        df = pd.read_csv(url)

        print("\n[Data Preview - First 5 Rows]")
        print(df.head())

        print("\n[Basic Statistics]")
        print(df.describe())

        print("\n[Class Distribution]")
        print(df["species"].value_counts())

        print("\nSuccess: Data downloaded and processed successfully!")

    except Exception as e:
        print(f"Error: Could not download data. {e}")
        sys.exit(1)  # Fail the workflow so GitHub Actions reports the error


if __name__ == "__main__":
    main()