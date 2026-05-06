"""
DataGen.py  —  Generates poker_dataset.csv for PokerML training.

All feature engineering is delegated to Features.py.
This file only handles: deck sampling, context randomisation, row assembly,
and CSV output.

Fixes applied
─────────────
#1  chen_score_normalized(...) was called with `...` (broken placeholder args).
    Now called correctly via build_feature_vector() in Features.py.
#2  calculate_ev was missing the `stack` param in some call sites, so
    ev_allin / best_ev keys were absent.  Features.calculate_ev always
    accepts stack (optional) and always returns best_ev + ev_allin.
#8  ev_dict spread into `row` now always has `best_ev` because it is
    guaranteed by Features.calculate_ev.
"""

import argparse
import csv
import random
from pathlib import Path

from treys import Card, Deck

# FIX #3 / #4: Import ALL feature engineering from Features.py.
# DataGen must use the same functions as app.py or training ≠ inference.
from Features import (
    OPPONENT_TYPES,
    POSITIONS,
    OPP_VPIP,
    OPP_AGGRESSION,
    build_feature_vector,
    decide_action,
)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Generate PokerML training dataset")
parser.add_argument("--rows",   type=int,   default=1_000_000, help="Number of rows to generate")
parser.add_argument("--out",    default="poker_dataset.csv", help="Output CSV path")
parser.add_argument("--seed",   type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)

# ── COLUMN ORDER ──────────────────────────────────────────────────────────────
# Must exactly match what ModelTrain.py expects.
# build_feature_vector returns all of these (minus _ev_allin which is
# stripped before writing).

COL_ORDER = [
    "rank1", "rank2", "suited", "pair", "gap", "chen_score",
    "stage", "position",
    "position_ratio", "seat_index", "num_players", "active_players",
    "stack_size", "pot_size", "to_call",
    "pot_odds", "spr", "effective_spr", "bet_ratio", "pot_commitment",
    "ev_call", "ev_raise", "ev_fold", "best_ev",
    "equity_edge",
    "hand_class", "hand_strength",
    "is_overpair", "is_top_pair", "is_middle_pair",
    "is_set", "is_two_pair", "is_straight", "is_flush", "hand_abstraction",
    "max_suit", "flush_draw", "flush_made", "board_flush_pressure",
    "board_connectedness", "board_paired", "board_high_card", "board_danger_score",
    "board_wetness",
    "straight_draw", "open_ended", "gutshot", "draw_type",
    "opp_type_encoded", "opp_vpip", "opp_aggression", "street_bet_pressure",
    "action",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

STAGE_BOARD_COUNT = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def sample_hand():
    """Return (hole_cards, board_cards, stage) with a fresh Deck each call."""
    deck  = Deck()
    deck.shuffle()
    stage = random.choice(list(STAGE_BOARD_COUNT.keys()))
    n     = STAGE_BOARD_COUNT[stage]
    hole  = [deck.draw(1)[0], deck.draw(1)[0]]
    board = [deck.draw(1)[0] for _ in range(n)]
    return hole, board, stage


def sample_context(players: int):
    """Random table context."""
    return {
        "position":    random.choice(POSITIONS),
        "opp_type":    random.choice(OPPONENT_TYPES),
        "stack":       random.uniform(10.0, 300.0),
        "pot":         random.uniform(2.0, 200.0),
        "to_call":     random.uniform(0.0, 80.0),
    }


# ── GENERATE ──────────────────────────────────────────────────────────────────

out_path = Path(args.out)
print(f"Generating {args.rows:,} rows → {out_path}")

with open(out_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=COL_ORDER)
    writer.writeheader()

    generated = 0
    attempts  = 0

    while generated < args.rows:
        attempts += 1
        try:
            players = random.randint(2, 9)
            hole, board, stage = sample_hand()
            ctx = sample_context(players)

            # FIX #1 / #3 / #4 / #5:
            # build_feature_vector is the ONE place chen_score_normalized,
            # get_hand_strength, calculate_ev (with stack) etc. are called.
            # No broken `...` placeholders; no copy-paste divergence.
            feat = build_feature_vector(
                hole     = hole,
                board    = board,
                stage    = stage,
                position = ctx["position"],
                stack    = ctx["stack"],
                pot      = ctx["pot"],
                to_call  = ctx["to_call"],
                players  = players,
                opp_type = ctx["opp_type"],
            )

            # ev_allin is a private UI helper — strip before deciding action / writing
            ev_allin = feat.pop("_ev_allin")   # may be None; only used by app.py

            # Derive heuristic label using values already computed in feat
            action = decide_action(
                hand_strength  = feat["hand_strength"],
                pot_odds       = feat["pot_odds"],
                spr            = feat["spr"],
                position       = ctx["position"],
                active_players = players,
                opponent_type  = ctx["opp_type"],
                ev_dict        = {          # FIX #2 / #8: all keys guaranteed present
                    "ev_call":  feat["ev_call"],
                    "ev_raise": feat["ev_raise"],
                    "best_ev":  feat["best_ev"],
                },
                wetness        = feat["board_wetness"],
                stack          = ctx["stack"],
            )

            row = {k: feat[k] for k in COL_ORDER if k != "action"}
            row["action"] = action

            writer.writerow(row)
            generated += 1

            if generated % 10_000 == 0:
                print(f"  {generated:,} / {args.rows:,}  (attempts: {attempts:,})")

        except Exception as exc:
            # Treys can raise on degenerate hands; just skip and retry
            continue

print(f"\nDone. {generated:,} rows written to {out_path}  ({attempts:,} attempts)")