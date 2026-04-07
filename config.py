# All magic numbers in one place — easy to tune for demo
RELEVANCE_WEIGHT = 0.6       # How much query relevance matters
FRESHNESS_WEIGHT = 0.4       # How much recency matters
DECAY_LAMBDA = 0.03          # Controls how fast freshness decays (per day)
                             # 0.03 = ~50% score at 23 days old
GPT_MODEL = "gpt-4o"
MAX_TOKENS_JUDGE = 200       # Keep latency low; we only need a score + reason
TOP_K = 3                    # Number of top chunks to highlight in UI
