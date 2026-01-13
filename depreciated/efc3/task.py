#!/usr/bin/env python3
import pandas as pd
import argparse
import time
import os


# Function to iterate through DataFrame and print each row
def print_coord(df):
    for index, row in df.iterrows():
        print(f"Trial: {row['TN']}, Coords: {row['Coords']}")
        print("Waiting for 10 seconds before continuing...")
        time.sleep(10)  # Wait for 10 seconds before prompting user input
        input("Press Enter to show the next row...")  # Wait for user input


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Print TMS locations from target file.")
    parser.add_argument('--target', type=str, default='efc3_coord_100.tsv', help="Target file")

    # Parse the arguments
    args = parser.parse_args()
    target_file = args.target

    target = pd.read_csv(os.path.join('target', target_file), sep='\t')

    print_coord(target)


if __name__ == "__main__":
    main()
