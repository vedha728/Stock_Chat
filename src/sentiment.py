import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Local FinBERT model (lazy-loaded on first call, cached after)
# ─────────────────────────────────────────────────────────────
_finbert_pipeline = None   # holds the loaded pipeline after first call
_finbert_load_attempted = False

def _load_finbert():
    """
    Loads ProsusAI/finbert locally using transformers pipeline.
    Model is downloaded once (~400MB) and cached by HuggingFace in
    the local cache directory (~/.cache/huggingface).
    Returns the pipeline object or None if loading fails.
    """
    global _finbert_pipeline, _finbert_load_attempted
    if _finbert_load_attempted:
        return _finbert_pipeline

    _finbert_load_attempted = True
    try:
        # Suppress noisy HuggingFace warnings before importing
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import pipeline, logging as hf_logging
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
        hf_logging.set_verbosity_error()   # suppress download progress noise

        print("[*] Loading FinBERT model locally (first run downloads ~400MB)...")
        t0 = time.time()
        _finbert_pipeline = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,
            truncation=True,
            max_length=512,
        )
        elapsed = time.time() - t0
        print(f"[+] FinBERT loaded in {elapsed:.1f}s. Running on CPU.")
        return _finbert_pipeline

    except ImportError:
        print("[Warning] transformers library not installed. Run: pip install transformers torch")
        return None
    except Exception as e:
        print(f"[Warning] Could not load local FinBERT: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Dictionary fallback (fast, dependency-free backup)
# ─────────────────────────────────────────────────────────────
BULLISH_WORDS = {
    "surge", "rise", "grow", "profit", "gain", "up", "bullish", "jump",
    "climb", "high", "positive", "beat", "upgrade", "success", "record",
    "exceed", "buy", "lead", "expand", "dividend", "rally", "outperform",
    "boost", "strong", "recovery", "acquisition", "win", "deal", "order"
}
BEARISH_WORDS = {
    "drop", "fall", "loss", "down", "bearish", "slump", "decline", "low",
    "negative", "miss", "downgrade", "fail", "plunge", "sink", "crash",
    "sell", "debt", "shrink", "investigation", "deficit", "warn", "weak",
    "cut", "layoff", "fraud", "penalty", "fine", "lawsuit", "risk", "threat"
}


def fallback_sentiment_analysis(headlines: list[str]) -> tuple[float, int, int]:
    """
    Fast dictionary-based sentiment analyzer. Used when FinBERT
    is unavailable. Returns (composite_score, positive_count, negative_count).
    """
    if not headlines:
        return 0.0, 0, 0

    scores    = []
    pos_count = 0
    neg_count = 0

    for headline in headlines:
        words    = set(headline.lower().replace(",", "").replace(".", "").split())
        pos_hits = len(words & BULLISH_WORDS)
        neg_hits = len(words & BEARISH_WORDS)

        if pos_hits > neg_hits:
            scores.append(0.5 + 0.1 * min(pos_hits, 5))
            pos_count += 1
        elif neg_hits > pos_hits:
            scores.append(-0.5 - 0.1 * min(neg_hits, 5))
            neg_count += 1
        else:
            scores.append(0.0)

    avg = sum(scores) / len(scores)
    return float(max(-1.0, min(1.0, avg))), pos_count, neg_count


def deduplicate_headlines(headlines: list[str]) -> list[str]:
    """
    Removes near-duplicate headlines to prevent sentiment inflation.
    Uses word-level Jaccard similarity (>65% overlap is considered a duplicate).
    """
    unique_headlines = []
    for h in headlines:
        h_clean = set(h.lower().replace(",", "").replace(".", "").replace("'", "").replace('"', "").split())
        if not h_clean:
            continue
        is_dup = False
        for uh in unique_headlines:
            uh_clean = set(uh.lower().replace(",", "").replace(".", "").replace("'", "").replace('"', "").split())
            intersection = len(h_clean & uh_clean)
            union = len(h_clean | uh_clean)
            if union == 0:
                continue
            similarity = intersection / union
            if similarity > 0.65:
                is_dup = True
                break
        if not is_dup:
            unique_headlines.append(h)
    return unique_headlines


# ─────────────────────────────────────────────────────────────
# Main entry point used by chatbot
# ─────────────────────────────────────────────────────────────
def analyze_news_sentiment(headlines: list[str]) -> tuple[float, int, int]:
    """
    Analyzes financial news headlines using a 2-tier strategy:
      Tier 1 — Local FinBERT (transformers, runs on CPU, no internet needed)
      Tier 2 — Dictionary-based fallback (instant, no dependencies)

    Returns: (composite_score [-1.0 to 1.0], positive_count, negative_count)
    """
    if not headlines:
        return 0.0, 0, 0

    # Remove near-duplicate headlines
    orig_count = len(headlines)
    headlines = deduplicate_headlines(headlines)
    if len(headlines) < orig_count:
        print(f"[*] De-duplicated headlines: kept {len(headlines)} of {orig_count} total headlines.")

    # ── Tier 1: Try local FinBERT ────────────────────────────
    pipe = _load_finbert()

    if pipe is not None:
        try:
            results   = pipe(headlines)
            scores    = []
            pos_count = 0
            neg_count = 0

            for prediction in results:
                # prediction is a list of {label, score} dicts for each headline
                label_scores = {item["label"].lower(): item["score"] for item in prediction}
                pos = label_scores.get("positive", 0.0)
                neg = label_scores.get("negative", 0.0)
                neu = label_scores.get("neutral",  0.0)

                if pos >= neg and pos >= neu:
                    scores.append(pos)
                    pos_count += 1
                elif neg >= pos and neg >= neu:
                    scores.append(-neg)
                    neg_count += 1
                else:
                    scores.append(0.0)

            composite = sum(scores) / len(scores) if scores else 0.0
            composite = float(max(-1.0, min(1.0, composite)))
            print(f"[+] FinBERT (local): score={composite:.2f} | pos={pos_count} | neg={neg_count}")
            return composite, pos_count, neg_count

        except Exception as e:
            print(f"[Warning] Local FinBERT inference error: {e}")

    # ── Tier 2: Dictionary fallback ──────────────────────────
    print("[*] Using dictionary-based sentiment fallback.")
    return fallback_sentiment_analysis(headlines)


if __name__ == "__main__":
    test_headlines = [
        "Tata Motors Q4 net profit jumps 222% to Rs 17,407 crore, beating estimates",
        "Wipro stock plunges 6% after weak revenue guidance",
        "TCS wins $500M mega deal in Europe",
        "Reliance Industries trades flat amid rangebound trading",
    ]
    print("Testing sentiment analysis on 4 headlines...\n")
    score, pos, neg = analyze_news_sentiment(test_headlines)
    print(f"\nFinal -> Composite Score: {score:.2f} | Positives: {pos} | Negatives: {neg}")
