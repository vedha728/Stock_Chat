"""
auto_retrain.py — Automated Retraining Trigger for StockChat AI Model

Usage:
  python auto_retrain.py              # Retrain only if model is >30 days old
  python auto_retrain.py --force      # Force retrain regardless of age
  python auto_retrain.py --status     # Check model age without retraining
  python auto_retrain.py --days 14    # Use custom staleness threshold (days)

How it works:
  - Reads training metadata from models/training_metadata.json
  - If model was trained more than 30 days ago (or --force is used), it runs the full
    training pipeline automatically
  - After training, updates the metadata file with the new timestamp
  - Safe to run daily via Windows Task Scheduler (see below)

Schedule via Windows Task Scheduler:
  Action: Start a program
  Program: python
  Arguments: D:\\Stockchat\\auto_retrain.py
  Start in: D:\\Stockchat
  Trigger: Daily at 2:00 AM (when market is closed and internet is stable)
"""

import os
import sys
import json
import argparse
import datetime
from datetime import timezone

# Add src/ to path so we can import train_model
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

METADATA_PATH = "models/training_metadata.json"
MODEL_PATH    = "models/xgboost_model.pkl"
DEFAULT_STALE_DAYS = 30   # Retrain if model is older than this many days


def load_metadata() -> dict:
    """Loads training metadata from JSON file. Returns empty dict if file doesn't exist."""
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[Warning] Could not read training_metadata.json — treating as fresh.")
    return {}


def save_metadata(data: dict):
    """Saves training metadata to JSON file."""
    os.makedirs("models", exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[+] Training metadata saved to {METADATA_PATH}")


def get_model_age_days(metadata: dict) -> float:
    """
    Returns the age of the current model in days.
    Falls back to reading the model file's modification time if metadata is unavailable.
    Returns infinity if neither is available.
    """
    # Try metadata first (most accurate)
    last_trained_str = metadata.get("last_trained_utc")
    if last_trained_str:
        try:
            last_trained = datetime.datetime.fromisoformat(last_trained_str)
            if last_trained.tzinfo is None:
                last_trained = last_trained.replace(tzinfo=timezone.utc)
            now = datetime.datetime.now(tz=timezone.utc)
            return (now - last_trained).total_seconds() / 86400.0
        except ValueError:
            pass

    # Fallback: check file modification time
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        file_dt = datetime.datetime.fromtimestamp(mtime, tz=timezone.utc)
        now = datetime.datetime.now(tz=timezone.utc)
        return (now - file_dt).total_seconds() / 86400.0

    return float('inf')   # No model exists at all


def print_status(metadata: dict, stale_days: int):
    """Prints a human-readable status of the current model."""
    print("\n" + "=" * 60)
    print("  STOCKCHAT MODEL STATUS")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print("  [ERROR] Model file: NOT FOUND")
        print(f"     Expected at: {MODEL_PATH}")
        print("     -> Run: python auto_retrain.py --force   to train now")
        print("=" * 60)
        return

    age_days = get_model_age_days(metadata)
    last_trained = metadata.get("last_trained_utc", "Unknown")
    num_stocks   = metadata.get("stocks_processed", "Unknown")
    num_rows     = metadata.get("training_rows", "Unknown")
    accuracy     = metadata.get("test_accuracy_pct", "Unknown")
    features     = metadata.get("feature_count", "Unknown")

    age_str = f"{age_days:.1f} days" if age_days != float('inf') else "Unknown"
    stale = age_days > stale_days

    print(f"  [FILE] Model file:       {MODEL_PATH}")
    print(f"  [DATE] Last trained:     {last_trained} (UTC)")
    print(f"  [AGE]  Model age:        {age_str}")
    print(f"  [DATA] Stocks used:      {num_stocks}")
    print(f"  [ROWS] Training rows:    {num_rows}")
    print(f"  [ACC]  Test accuracy:    {accuracy}")
    print(f"  [FEAT] Feature count:    {features}")
    print(f"  [THR]  Stale threshold:  {stale_days} days")
    print()

    if stale:
        print(f"  [WARN] STATUS: STALE -- model is >{stale_days} days old.")
        print(f"     -> Run: python auto_retrain.py   to retrain automatically")
        print(f"     -> Run: python auto_retrain.py --force   to retrain immediately")
    else:
        remaining = stale_days - age_days
        print(f"  [OK] STATUS: FRESH -- model is up-to-date.")
        print(f"     -> Next scheduled retrain in ~{remaining:.0f} days")

    print("=" * 60 + "\n")


def run_retrain() -> dict:
    """
    Runs the full training pipeline and returns metadata about the run.
    Imports train_model at runtime to avoid circular import issues.
    """
    print("\n" + "=" * 60)
    print("  STARTING AUTOMATED RETRAINING")
    print("=" * 60 + "\n")

    start_time = datetime.datetime.now(tz=timezone.utc)

    # Import and run training pipeline
    from train_model import run_training_pipeline
    metrics = run_training_pipeline() or {}

    end_time = datetime.datetime.now(tz=timezone.utc)
    duration = (end_time - start_time).total_seconds()

    print(f"\n[+] Retraining completed in {duration:.1f} seconds.")

    # Try to read model stats for metadata
    feature_count = None
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        feature_count = getattr(model, 'n_features_in_', None)
    except Exception:
        pass

    ret = {
        "last_trained_utc": end_time.isoformat(),
        "feature_count": feature_count,
        "training_duration_seconds": round(duration, 1),
    }
    ret.update(metrics)
    return ret


def main():
    parser = argparse.ArgumentParser(
        description="Automated retraining trigger for StockChat AI model."
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force retrain regardless of model age."
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show model age and status without retraining."
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_STALE_DAYS,
        help=f"Number of days before model is considered stale (default: {DEFAULT_STALE_DAYS})."
    )
    args = parser.parse_args()

    # Load existing metadata
    metadata = load_metadata()

    # --status: just report, don't retrain
    if args.status:
        print_status(metadata, args.days)
        return

    # Determine whether to retrain
    age_days  = get_model_age_days(metadata)
    is_stale  = age_days > args.days
    no_model  = not os.path.exists(MODEL_PATH)

    print_status(metadata, args.days)

    if args.force:
        print("[*] --force flag detected. Retraining now regardless of model age...")
        should_retrain = True
    elif no_model:
        print("[*] No model file found. Training from scratch...")
        should_retrain = True
    elif is_stale:
        print(f"[*] Model is {age_days:.1f} days old (threshold: {args.days} days). Retraining now...")
        should_retrain = True
    else:
        remaining = args.days - age_days
        print(f"[OK] Model is fresh ({age_days:.1f} days old). No retrain needed.")
        print(f"    Next auto-retrain in ~{remaining:.0f} days.")
        print("    Use --force to retrain immediately.\n")
        should_retrain = False

    if should_retrain:
        retrain_info = run_retrain()

        # Save only the fresh metrics from this run to avoid any stale leftovers
        save_metadata(retrain_info)

        print("\n" + "=" * 60)
        print("  [OK] RETRAINING COMPLETE")
        print(f"  Trained at: {retrain_info['last_trained_utc']} (UTC)")
        print(f"  Duration:   {retrain_info['training_duration_seconds']}s")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
