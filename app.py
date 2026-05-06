import os
import gdown
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from treys import Card

# FIX #3 / #5: Import ALL feature engineering from the single canonical module.
# Nothing is copy-pasted here — app.py is pure UI + model loading.
from Features import (
    build_feature_vector,
    OPPONENT_TYPES,
    OPP_VPIP,
    OPP_AGGRESSION,
)

# ── MODEL DOWNLOAD ────────────────────────────────────────────────────────────
FOLDER_URL = "https://drive.google.com/drive/folders/1W6gHiQ2SnTJT7l77JgiBW5we-v2v-LOz"

if not os.path.exists("poker_model"):
    gdown.download_folder(FOLDER_URL, quiet=False, use_cookies=False)

st.title("Poker Decision Advisor")

# ── MODEL LOAD ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_dir="poker_model"):
    p    = Path(model_dir)
    meta = json.load(open(p / "meta.json"))
    les  = joblib.load(p / "label_encoders.pkl")
    sc   = joblib.load(p / "scaler.pkl")
    rf   = joblib.load(p / "rf_model.pkl")
    return meta, les, sc, rf


def predict(hand, meta, les, sc, rf):
    row = pd.DataFrame([hand])
    for col in meta["cat_cols"]:
        row[col] = les[col].transform(row[col])
    X      = sc.transform(row[meta["feature_cols"]].values.astype(float))
    proba  = rf.predict_proba(X)[0]
    action = les["action"].inverse_transform([proba.argmax()])[0]
    return action.upper(), dict(zip(meta["class_names"], proba))


meta, les, sc, rf = load_model()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
RANK_LABELS = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}

# treys rank/suit strings (UI display only — never used for card_suit() logic)
TREYS_RANK = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
              10:"T",11:"J",12:"Q",13:"K",14:"A"}
TREYS_SUIT = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}

SUIT_OPTIONS = ["♠", "♥", "♦", "♣"]

OPP_LABELS = {
    "tight_passive":    "Tight Passive",
    "tight_aggressive": "Tight Aggressive",
    "loose_passive":    "Loose Passive",
    "loose_aggressive": "Loose Aggressive",
}

# ── CARD HELPER (UI → treys int) ──────────────────────────────────────────────
def make_treys_card(rank_int: int, suit_sym: str) -> int:
    """Convert (14, '♠') → treys Card int."""
    r = TREYS_RANK[rank_int]
    s = TREYS_SUIT[suit_sym]
    return Card.new(r + s)


# ── UI ────────────────────────────────────────────────────────────────────────

st.header("Hole Cards")
col1, col2 = st.columns(2)
with col1:
    rank1 = st.selectbox("Card 1 Rank", list(RANK_LABELS.keys()),
                         format_func=lambda x: RANK_LABELS[x], index=12)  # default A
    suit1 = st.selectbox("Card 1 Suit", SUIT_OPTIONS)
with col2:
    rank2 = st.selectbox("Card 2 Rank", list(RANK_LABELS.keys()),
                         format_func=lambda x: RANK_LABELS[x], index=11)  # default K
    suit2 = st.selectbox("Card 2 Suit", SUIT_OPTIONS, index=1)

# duplicate card guard
if rank1 == rank2 and suit1 == suit2:
    st.error("Both hole cards are identical — please fix before continuing.")
    st.stop()

st.header("Situation")
col3, col4 = st.columns(2)
with col3:
    stage = st.selectbox("Street", ["preflop", "flop", "turn", "river"])
with col4:
    position = st.selectbox("Position", ["early", "middle", "late", "button"])

players = st.number_input("Players at Table", 2, 9, 6)

opp_label = st.selectbox("Opponent Type", list(OPP_LABELS.values()))
opp_type  = [k for k, v in OPP_LABELS.items() if v == opp_label][0]

st.header("Stack & Pot")
stack   = st.number_input("Stack Size (BB)", 1, 500, 100)
pot     = st.number_input("Pot Size (BB)",   1, 1000, 20)
to_call = st.number_input("To Call (BB)",    0, 500,  8)

# ── BOARD CARDS ───────────────────────────────────────────────────────────────
board_cards = []

if stage != "preflop":
    st.header("Board Cards")

    n_board     = {"flop": 3, "turn": 4, "river": 5}[stage]
    board_cols  = st.columns(n_board)
    board_inputs = []  # list of (rank_int, suit_str)

    for i, bc in enumerate(board_cols):
        with bc:
            st.markdown(f"**Card {i + 1}**")
            br = st.selectbox("Rank", list(RANK_LABELS.keys()),
                              format_func=lambda x: RANK_LABELS[x],
                              key=f"br{i}", index=i)
            bs = st.selectbox("Suit", SUIT_OPTIONS,
                              key=f"bs{i}", index=i % 4)
            board_inputs.append((br, bs))

    # build treys ints; validate no dupes
    hole_set    = {(rank1, suit1), (rank2, suit2)}
    board_set   = set()
    valid_board = True

    for br, bs in board_inputs:
        if (br, bs) in hole_set or (br, bs) in board_set:
            st.error(f"Duplicate card detected: {RANK_LABELS[br]}{bs} — fix before continuing.")
            valid_board = False
            break
        board_set.add((br, bs))
        board_cards.append(make_treys_card(br, bs))

    if not valid_board:
        st.stop()

# ── ANALYZE ───────────────────────────────────────────────────────────────────
if st.button("Analyze Hand", type="primary"):

    hole = [make_treys_card(rank1, suit1), make_treys_card(rank2, suit2)]

    # FIX #3 / #5: call the canonical build_feature_vector from Features.py.
    # The old inline build_hand() is gone — one source of truth.
    hand = build_feature_vector(
        hole        = hole,
        board       = board_cards,
        stage       = stage,
        position    = position,
        stack       = float(stack),
        pot         = float(pot),
        to_call     = float(to_call),
        players     = int(players),
        opp_type    = opp_type,
    )

    # pull out ev_allin before feeding to model (not a model feature)
    ev_allin = hand.pop("_ev_allin")

    missing = [f for f in meta["feature_cols"] if f not in hand]
    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    action, probs = predict(hand, meta, les, sc, rf)

    # ── ACTION ────────────────────────────────────────────────────────────────
    st.header("Recommended Action")
    action_color = {
        "FOLD":   "🔴",
        "CALL":   "🟡",
        "RAISE":  "🟢",
        "ALL-IN": "🔥",
    }
    emoji = action_color.get(action, "🃏")
    st.success(f"{emoji}  {action}")

    # ── PROBABILITIES ─────────────────────────────────────────────────────────
    st.subheader("Action Probabilities")
    for k, v in sorted(probs.items(), key=lambda x: -x[1]):
        st.progress(float(v), text=f"{k.upper()}: {v * 100:.1f}%")

    # ── EV ────────────────────────────────────────────────────────────────────
    st.subheader("EV Estimates")
    ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
    ev_col1.metric("Raise EV",  round(hand["ev_raise"], 3))
    ev_col2.metric("Call EV",   round(hand["ev_call"],  3))
    ev_col3.metric("Fold EV",   0)
    ev_col4.metric("All-In EV", round(ev_allin, 3) if ev_allin is not None else "N/A")

    # ── HAND DETAILS ──────────────────────────────────────────────────────────
    with st.expander("Hand Details"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Chen Score:** {hand['chen_score']:.3f}")
            st.write(f"**Hand Strength (adjusted):** {hand['hand_strength']:.3f}")
            st.write(f"**Pot Odds:** {hand['pot_odds']:.3f}")
            st.write(f"**Equity Edge:** {hand['equity_edge']:.3f}")
            st.write(f"**SPR:** {hand['spr']:.2f}")
            st.write(f"**Suited:** {'Yes' if hand['suited'] else 'No'}")
        with col_b:
            hc_names = {
                1: "Straight Flush", 2: "Quads", 3: "Full House",
                4: "Flush", 5: "Straight", 6: "Set",
                7: "Two Pair", 8: "One Pair", 9: "High Card",
            }
            st.write(f"**Hand Class:** {hc_names.get(hand['hand_class'], '?')}")
            st.write(f"**Board Wetness:** {hand['board_wetness']}/3")
            st.write(f"**Flush Draw:** {'Yes' if hand['flush_draw'] else 'No'}")
            st.write(f"**Straight Draw:** {'Yes' if hand['straight_draw'] else 'No'}")
            st.write(f"**Open-Ended:** {'Yes' if hand['open_ended'] else 'No'}")
            st.write(f"**Gutshot:** {'Yes' if hand['gutshot'] else 'No'}")

        if stage != "preflop":
            st.divider()
            st.write("**Board Texture**")
            bcol1, bcol2, bcol3 = st.columns(3)
            bcol1.metric("Connectedness", hand["board_connectedness"])
            bcol2.metric("Paired",        hand["board_paired"])
            bcol3.metric("Danger Score",  hand["board_danger_score"])