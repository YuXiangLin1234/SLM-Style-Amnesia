import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Compute non-neutral emotion-prediction accuracy from a probabilities CSV."
    )
    parser.add_argument(
        "csv_path",
        help="CSV with prob_* columns and a gold_label column.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    print(f"[Log] Found probability columns: {prob_cols}")

    prob_cols_no_neutral = [c for c in prob_cols if c != "prob_Neutral"]
    print(f"[Log] Using columns (excluding Neutral): {prob_cols_no_neutral}")

    label_names = [c.replace("prob_", "") for c in prob_cols_no_neutral]
    probs = df[prob_cols_no_neutral].to_numpy()
    pred_labels_new = [label_names[idx] for idx in probs.argmax(axis=1)]
    df["pred_label_no_neutral"] = pred_labels_new

    df_eval = df[df["gold_label"] != "Neutral"].copy()
    df_eval["correct_no_neutral"] = (
        df_eval["gold_label"].str.lower() == df_eval["pred_label_no_neutral"].str.lower()
    )

    accuracy = df_eval["correct_no_neutral"].mean()
    total = len(df_eval)
    correct = df_eval["correct_no_neutral"].sum()

    print(f"Total evaluated (excluding gold=Neutral): {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy (excluding Neutral & using prob_ columns): {accuracy:.4f}")


if __name__ == "__main__":
    main()
