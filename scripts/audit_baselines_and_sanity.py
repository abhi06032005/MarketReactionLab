"""Audit Script 3: Majority-class baseline, shuffled-label sanity check, seed variability, and replay leaks."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from src.models.evaluation import evaluate_model
from src.models.baseline_price import extract_features_and_labels
from src.replay.historical_store import HistoricalReplayStore

def run_audit_baselines_and_sanity():
    print("=== CHECK 10: Majority-Class Baseline & Shuffled-Label Sanity Check ===")
    train_path = "data/processed/train.parquet"
    test_path = "data/processed/test.parquet"

    X_train, y_train, _, _ = extract_features_and_labels(train_path)
    X_test, y_test, ret_test, _ = extract_features_and_labels(test_path)

    # 1. Majority-class baseline
    majority_class = int(np.bincount(y_train).argmax())
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_majority.fit(X_train, y_train)
    y_pred_maj = dummy_majority.predict(X_test)
    res_maj = evaluate_model("Majority-Class Baseline", y_test, y_pred_maj, ret_test)

    print(f"Majority class in training set: {majority_class} (DOWN=0, NEUTRAL=1, UP=2)")
    print(f"Majority-Class Baseline Accuracy: {res_maj['classification']['accuracy']:.4f}")
    print(f"Majority-Class Balanced Accuracy: {res_maj['classification']['balanced_accuracy']:.4f}")
    print(f"Majority-Class Macro F1: {res_maj['classification']['macro_f1']:.4f}")
    print(f"Majority-Class MCC: {res_maj['classification']['mcc']:.4f}")

    # 2. Uniform random guessing baseline
    dummy_uniform = DummyClassifier(strategy="uniform", random_state=42)
    dummy_uniform.fit(X_train, y_train)
    y_pred_rand = dummy_uniform.predict(X_test)
    res_rand = evaluate_model("Uniform Random Baseline", y_test, y_pred_rand, ret_test)
    print(f"Uniform Random Baseline Accuracy: {res_rand['classification']['accuracy']:.4f}")
    print(f"Uniform Random Macro F1: {res_rand['classification']['macro_f1']:.4f}")

    # 3. Shuffled-Label Sanity Check
    # If a model can predict labels when y_train is randomly permuted, it indicates data leakage or bug!
    print("\nRunning Shuffled-Label Sanity Check (Permuting training labels)...")
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)

    clf_shuffled = HistGradientBoostingClassifier(random_state=42, max_iter=100)
    clf_shuffled.fit(X_train, y_train_shuffled)
    y_pred_shuffled = clf_shuffled.predict(X_test)
    res_shuffled = evaluate_model("Shuffled-Label Check (HistGBM)", y_test, y_pred_shuffled, ret_test)

    print(f"Shuffled-Label Accuracy: {res_shuffled['classification']['accuracy']:.4f}")
    print(f"Shuffled-Label Macro F1: {res_shuffled['classification']['macro_f1']:.4f}")
    print(f"Shuffled-Label MCC: {res_shuffled['classification']['mcc']:.4f}")
    print(f"Shuffled-Label Sharpe: {res_shuffled['financial_simulation']['annualized_sharpe']:.3f}")

    print("\n=== CHECK 11: Fixed Seeds Variability (HistGBM across 5 seeds) ===")
    seeds = [1, 42, 123, 777, 2026]
    accs, f1s, sharpes = [], [], []
    for s in seeds:
        clf = HistGradientBoostingClassifier(random_state=s, max_iter=150, learning_rate=0.05)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        r = evaluate_model(f"HistGBM-seed-{s}", y_test, pred, ret_test)
        accs.append(r["classification"]["accuracy"])
        f1s.append(r["classification"]["macro_f1"])
        sharpes.append(r["financial_simulation"]["annualized_sharpe"])

    print(f"Seeds: {seeds}")
    print(f"Accuracies: {accs} -> Mean: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"Macro F1s:  {f1s} -> Mean: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"Sharpe:     {sharpes} -> Mean: {np.mean(sharpes):.3f} +/- {np.std(sharpes):.3f}")

    print("\n=== CHECK 9: Tool-Enabled Historical Replay Future Date Leak Checks ===")
    store = HistoricalReplayStore()
    
    # Test 1: Query price history with as_of 2015-01-01
    as_of_test = "2015-01-01"
    prices = store.get_price_history("AAPL", as_of=as_of_test, lookback_days=100)
    future_prices = [p for p in prices if p["date"] > as_of_test]
    print(f"Price query as-of {as_of_test}: returned {len(prices)} candles, future leaks: {len(future_prices)}")

    # Test 2: Query news history with as_of 2015-01-01
    news = store.get_historical_news("AAPL", as_of=as_of_test, lookback_days=100)
    future_news = [n for n in news if n["calendar_date"] > as_of_test]
    print(f"News query as-of {as_of_test}: returned {len(news)} articles, future leaks: {len(future_news)}")

    # Test 3: Similar events outcome censorship
    events = store.find_similar_events("AAPL", as_of="2018-05-15", top_k=10)
    uncensored_future_outcomes = [
        ev for ev in events 
        if ev["target_session"] > "2018-05-15" and ev["outcome_status"] != "CENSORED_FUTURE_OUTCOME"
    ]
    print(f"Similar events query as-of 2018-05-15: uncensored future leaks: {len(uncensored_future_outcomes)}")

if __name__ == "__main__":
    run_audit_baselines_and_sanity()
