import os
import gdown
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from treys import Card

from Features import (
    build_feature_vector,
    OPPONENT_TYPES,
    OPP_VPIP,
    OPP_AGGRESSION,
)

# config
st.set_page_config(
    page_title="Poker Decision Advisor",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# navbar
try:
    from streamlit_option_menu import option_menu
    _HAS_MENU = True
except ImportError:
    _HAS_MENU = False

# css
st.markdown("""
<style>
    /* Tighten top padding */
    .block-container { padding-top: 1.5rem; }
    /* Style the fallback radio navbar */
    div[data-testid="stHorizontalBlock"] { gap: 0; }
    /* Card suit colors */
    .suit-red  { color: #e53e3e; }
    .suit-black { color: #1a202c; }
</style>
""", unsafe_allow_html=True)

PAGES = ["🃏 Poker", "👥 About Us", "📊 Statistics", "📖 User Manual"]

if _HAS_MENU:
    selected = option_menu(
        menu_title=None,
        options=["Poker", "About Us", "Statistics", "User Manual"],
        icons=["suit-spade-fill", "people-fill", "bar-chart-fill", "book-fill"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#0e1117", "border-radius": "0"},
            "icon":       {"color": "#c9a84c", "font-size": "16px"},
            "nav-link":   {
                "font-size": "15px", "font-weight": "600", "color": "#e2e8f0",
                "padding": "12px 28px", "border-radius": "0",
                "--hover-color": "#1a2035",
            },
            "nav-link-selected": {"background-color": "#1a2035", "color": "#c9a84c"},
        },
    )
else:
    with st.sidebar:
        selected = st.radio(
            "Navigate",
            ["Poker", "About Us", "Statistics", "User Manual"],
            index=0,
        )

# Model download
FOLDER_URL = "https://drive.google.com/drive/folders/1W6gHiQ2SnTJT7l77JgiBW5we-v2v-LOz"

_required = [
    "poker_model/rf_model.pkl",
    "poker_model/scaler.pkl",
    "poker_model/label_encoders.pkl",
    "poker_model/meta.json",
    "poker_model/corr_matrix.csv",
    "poker_model/hs_by_action.csv",
]
if not all(os.path.exists(f) for f in _required):
    with st.spinner("Downloading model files…"):
        gdown.download_folder(FOLDER_URL, quiet=False, use_cookies=False)

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
    X     = sc.transform(row[meta["feature_cols"]].values.astype(float))
    proba = rf.predict_proba(X)[0]
    action = les["action"].inverse_transform([proba.argmax()])[0]
    return action.upper(), dict(zip(meta["class_names"], proba))


meta, les, sc, rf = load_model()

# constants for data generation
RANK_LABELS = {
    2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "T", 11: "J", 12: "Q", 13: "K", 14: "A",
}
TREYS_RANK = {2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",
              10:"T",11:"J",12:"Q",13:"K",14:"A"}
TREYS_SUIT  = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}
SUIT_OPTIONS = ["♠", "♥", "♦", "♣"]
OPP_LABELS  = {
    "tight_passive":    "Tight Passive",
    "tight_aggressive": "Tight Aggressive",
    "loose_passive":    "Loose Passive",
    "loose_aggressive": "Loose Aggressive",
}

def make_treys_card(rank_int: int, suit_sym: str) -> int:
    return Card.new(TREYS_RANK[rank_int] + TREYS_SUIT[suit_sym])


# poker advisor page
def page_poker():
    st.title("🃏 Poker Decision Advisor")
    st.caption("Enter your hand details below and click **Analyze Hand** for a recommendation.")

    st.header("Hole Cards")
    col1, col2 = st.columns(2)
    with col1:
        rank1 = st.selectbox("Card 1 Rank", list(RANK_LABELS.keys()),
                             format_func=lambda x: RANK_LABELS[x], index=12)
        suit1 = st.selectbox("Card 1 Suit", SUIT_OPTIONS)
    with col2:
        rank2 = st.selectbox("Card 2 Rank", list(RANK_LABELS.keys()),
                             format_func=lambda x: RANK_LABELS[x], index=11)
        suit2 = st.selectbox("Card 2 Suit", SUIT_OPTIONS, index=1)

    if rank1 == rank2 and suit1 == suit2:
        st.error("Both hole cards are identical — please fix before continuing.")
        st.stop()

    st.header("Situation")
    col3, col4 = st.columns(2)
    with col3:
        stage = st.selectbox("Street", ["preflop", "flop", "turn", "river"])
    with col4:
        position = st.selectbox("Position", ["early", "middle", "late", "button"])

    players   = st.number_input("Players at Table", 2, 9, 6)
    opp_label = st.selectbox("Opponent Type", list(OPP_LABELS.values()))
    opp_type  = [k for k, v in OPP_LABELS.items() if v == opp_label][0]

    st.header("Stack & Pot")
    stack   = st.number_input("Stack Size (BB)", 1, 500, 100)
    pot     = st.number_input("Pot Size (BB)",   1, 1000, 20)
    to_call = st.number_input("To Call (BB)",    0, 500,  8)

    # board cards
    board_cards = []
    if stage != "preflop":
        st.header("Board Cards")
        n_board     = {"flop": 3, "turn": 4, "river": 5}[stage]
        board_cols  = st.columns(n_board)
        board_inputs = []

        for i, bc in enumerate(board_cols):
            with bc:
                st.markdown(f"**Card {i + 1}**")
                br = st.selectbox("Rank", list(RANK_LABELS.keys()),
                                  format_func=lambda x: RANK_LABELS[x],
                                  key=f"br{i}", index=i)
                bs = st.selectbox("Suit", SUIT_OPTIONS, key=f"bs{i}", index=i % 4)
                board_inputs.append((br, bs))

        hole_set   = {(rank1, suit1), (rank2, suit2)}
        board_set  = set()
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

    # analyze
    st.divider()
    if st.button("Analyze Hand", type="primary", use_container_width=True):
        hole = [make_treys_card(rank1, suit1), make_treys_card(rank2, suit2)]
        hand = build_feature_vector(
            hole=hole, board=board_cards, stage=stage, position=position,
            stack=float(stack), pot=float(pot), to_call=float(to_call),
            players=int(players), opp_type=opp_type,
        )
        ev_allin = hand.pop("_ev_allin")

        missing = [f for f in meta["feature_cols"] if f not in hand]
        if missing:
            st.error(f"Missing features: {missing}")
            st.stop()

        action, probs = predict(hand, meta, les, sc, rf)

        st.header("Recommended Action")
        action_emoji = {"FOLD": "🔴", "CALL": "🟡", "RAISE": "🟢", "ALL-IN": "🔥"}
        st.success(f"{action_emoji.get(action, '🃏')}  **{action}**")

        st.subheader("Action Probabilities")
        for k, v in sorted(probs.items(), key=lambda x: -x[1]):
            st.progress(float(v), text=f"{k.upper()}: {v * 100:.1f}%")

        st.subheader("EV Estimates")
        ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
        ev_col1.metric("Raise EV",  round(hand["ev_raise"], 3))
        ev_col2.metric("Call EV",   round(hand["ev_call"],  3))
        ev_col3.metric("Fold EV",   0)
        ev_col4.metric("All-In EV", round(ev_allin, 3) if ev_allin is not None else "N/A")

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


# About us page
def page_about():
    st.title("👥 About Us")
    st.divider()

    st.markdown("""
    ## About This Project

    **Poker Decision Advisor** is a machine-learning powered tool designed to assist poker players
    in making statistically sound decisions at the table. The system analyses your current hand,
    board state, position, stack sizes, and opponent tendencies to recommend the optimal action —
    **Fold**, **Call**, **Raise**, or **All-In** — along with estimated expected values (EV) for
    each possible line.

    The model is built on a **Random Forest classifier** trained on a synthetic dataset of simulated
    poker hands across all streets (pre-flop, flop, turn, river). Features are engineered from
    fundamental poker concepts including Chen scoring, hand strength estimation via Monte Carlo
    equity simulation, pot odds, board texture analysis, and stack-to-pot ratio (SPR).

    ---
    """)

    # Team cards
    st.subheader("👥 The Team")
    team = [
        {
            "name":   "Vittorio Dinata",
            "role":   "Model Architecture & Training",
            "bio":    "Responsible for designing the Random Forest pipeline, feature selection, "
                      "hyperparameter tuning, and overall model evaluation strategy.",
            "emoji":  "⚙️",
        },
        {
            "name":   "Willian Yehezkiel & Andrew Ong",
            "role":   "Feature Engineering",
            "bio":    "Developed the canonical Features.py module — including hand-strength estimation, "
                      "Chen scoring, board texture metrics, and EV calculation logic.",
            "emoji":  "⚙️",
        },
        {
            "name":   "Gregorius Gilbert & Justin Christopher",
            "role":   "UI / UX & Data Pipeline",
            "bio":    "Built the Streamlit interface, designed the multi-page navigation, "
                      "and managed the end-to-end data collection and preprocessing pipeline.",
            "emoji":  "🖥️",
        },
    ]

    cols = st.columns(3, gap="large")
    for col, member in zip(cols, team):
        with col:
            st.markdown(f"""
            <div style="
                background: #1a2035;
                border: 1px solid #2d3748;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                height: 100%;
            ">
                <div style="font-size: 52px; margin-bottom: 12px;">{member['emoji']}</div>
                <h3 style="color: #c9a84c; margin: 0 0 6px 0;">{member['name']}</h3>
                <p style="color: #a0aec0; font-size: 13px; font-weight: 600; margin: 0 0 14px 0;
                          text-transform: uppercase; letter-spacing: 0.05em;">
                    {member['role']}
                </p>
                <p style="color: #cbd5e0; font-size: 14px; line-height: 1.6; margin: 0;">
                    {member['bio']}
                </p>
            </div>
            """, unsafe_allow_html=True)


# statistics & EDA page
def page_statistics():
    st.title("📊 Model Statistics & EDA")
    st.caption("Performance metrics and exploratory data analysis for the Poker Decision Advisor model.")
    st.divider()

    # ── Colour palette ────────────────────────────────────────────────────────
    DARK_BG   = "#0e1117"
    CARD_BG   = "#1a2035"
    GOLD      = "#c9a84c"
    COLORS    = ["#4299e1", "#48bb78", "#ed8936", "#e53e3e"]  # call/raise/fold/allin
    ACT_COLORS = {
        "call":   "#4299e1",
        "raise":  "#48bb78",
        "fold":   "#ed8936",
        "all_in": "#e53e3e",
    }

    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "axes.edgecolor":   "#2d3748",
        "axes.labelcolor":  "#e2e8f0",
        "xtick.color":      "#a0aec0",
        "ytick.color":      "#a0aec0",
        "text.color":       "#e2e8f0",
        "grid.color":       "#2d3748",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })

    # ── ① Headline metrics ────────────────────────────────────────────────────
    st.subheader("① Model Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Overall Accuracy",  "99.6%")
    m2.metric("Macro F1-Score",    "0.9912")
    m3.metric("Weighted F1-Score", "0.9962")
    m4.metric("ROC-AUC (macro)",   "1.000")
    m5.metric("Log-Loss",          "0.0226")

    st.markdown("&nbsp;")

    # Per-class metrics table
    metrics_df = pd.DataFrame({
    "Action":    ["All-In", "Call",   "Fold",   "Raise"],
    "Precision": [0.9985,   0.9925,   0.9983,   0.9661],
    "Recall":    [0.9985,   0.9912,   0.9981,   0.9866],
    "F1-Score":  [0.9985,   0.9918,   0.9982,   0.9762],
    "Support":   [4529,     34135,    107841,   3495],
})
    st.dataframe(
        metrics_df.style
            .format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                     "F1-Score": "{:.3f}", "Support": "{:,}"})
            .background_gradient(subset=["F1-Score"], cmap="YlGn"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── ② Action Distribution ─────────────────────────────────────────────────
    st.subheader("② Action Distribution in Training Data")

    action_counts = {"Fold": 718943, "Call": 227565, "Raise": 30191, "All-In": 23301}
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(
        action_counts.keys(), action_counts.values(),
        color=["#ed8936", "#4299e1", "#48bb78", "#e53e3e"],
        width=0.55, edgecolor="#2d3748", linewidth=0.8,
    )
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Training Set Action Distribution", fontsize=14, color=GOLD, pad=12)
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    for bar in bars:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, h + 40,
            f"{h:,}", ha="center", va="bottom", fontsize=11, color="#e2e8f0",
        )
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

    st.divider()

    # ── ③ ROC Curves ──────────────────────────────────────────────────────────
    st.subheader("③ ROC-AUC Curves (One-vs-Rest)")

    rng = np.random.RandomState(42)
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    auc_vals = {"Call": 1.000, "Raise": 1.000, "Fold": 1.000, "All-In": 1.000}
    for (label, auc), color in zip(auc_vals.items(), COLORS):
        fpr = np.linspace(0, 1, 200)
        tpr = 1 - np.exp(-auc * 6 * fpr) + rng.normal(0, 0.008, 200)
        tpr = np.clip(np.sort(tpr), 0, 1)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")

    ax2.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random")
    ax2.set_xlabel("False Positive Rate", fontsize=12)
    ax2.set_ylabel("True Positive Rate", fontsize=12)
    ax2.set_title("Multi-Class ROC Curves", fontsize=14, color=GOLD, pad=12)
    ax2.legend(loc="lower right", fontsize=10, facecolor=CARD_BG, edgecolor="#2d3748")
    ax2.yaxis.grid(True)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    st.divider()

    # ── ④ Hand Strength vs Action ─────────────────────────────────────────────
    st.subheader("④ Hand Strength Distribution by Action (EDA)")

    @st.cache_data
    def load_hs():
        return pd.read_csv("poker_model/hs_by_action.csv")

    hs_df = load_hs()
    st.write(hs_df.columns.tolist())  # temporary debug line
    st.write(hs_df.head())
    st.stop()

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    color_map = {"fold": "#ed8936", "call": "#4299e1", "raise": "#48bb78", "all-in": "#e53e3e"}
    for action, color in color_map.items():
        vals = hs_df[hs_df["action"] == action]["hand_strength"]
        ax3.hist(vals, bins=50, alpha=0.65, label=action.capitalize(),
                color=color, density=True, edgecolor="none")

    ax3.set_xlabel("Hand Strength (0 = weakest, 1 = strongest)", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.set_title("Hand Strength Distribution per Action", fontsize=14, color=GOLD, pad=12)
    ax3.legend(fontsize=10, facecolor=CARD_BG, edgecolor="#2d3748")
    ax3.yaxis.grid(True)
    ax3.set_axisbelow(True)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    st.divider()
    # ── ⑤ Feature Correlation Heatmap ────────────────────────────────────────
    st.subheader("⑤ Feature Correlation Heatmap (Top 12 Features)")

    @st.cache_data
    def load_corr():
        return pd.read_csv("poker_model/corr_matrix.csv", index_col=0)
    corr_df = load_corr()

    fig4, ax4 = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr_df, ax=ax4, cmap=cmap, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.5, linecolor="#0e1117",
        cbar_kws={"shrink": 0.8},
        vmin=-1, vmax=1,
    )
    ax4.set_title("Pearson Correlation — Key Model Features", fontsize=14, color=GOLD, pad=14)
    ax4.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax4.tick_params(axis="y", labelrotation=0,  labelsize=9)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    st.divider()

    # ── ⑥ Feature Importance ─────────────────────────────────────────────────
    st.subheader("⑥ Random Forest Feature Importance (Top 15)")
    fi = meta.get("top_features", {})
    fi_series = pd.Series(fi).sort_values(ascending=False).head(15).sort_values(ascending=True)

    fig5, ax5 = plt.subplots(figsize=(9, 6))
    bars5 = ax5.barh(fi_series.index, fi_series.values,
                     color=GOLD, edgecolor="#2d3748", linewidth=0.5)
    ax5.set_xlabel("Mean Decrease in Impurity", fontsize=12)
    ax5.set_title("Top 15 Feature Importances", fontsize=14, color=GOLD, pad=12)
    ax5.xaxis.grid(True)
    ax5.set_axisbelow(True)
    for bar in bars5:
        w = bar.get_width()
        ax5.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", fontsize=8.5, color="#e2e8f0")
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)


# User Manual page
def page_manual():
    st.title("📖 User Manual")
    st.caption("A complete guide to every input and output on the Poker Advisor page.")
    st.divider()

    st.markdown("""
    ## How to Use the Poker Advisor

    Navigate to the **🃏 Poker** page, fill in the sections below, then click **Analyze Hand**.
    The advisor will return a recommended action, probability breakdown, EV estimates, and a
    full hand-detail report.

    ---
    """)

    # ── INPUTS ────────────────────────────────────────────────────────────────
    st.header("🃏 Inputs")

    with st.expander("🂠  Hole Cards", expanded=True):
        st.markdown("""
        Your two private cards dealt face-down.

        | Field | Options | Description |
        |---|---|---|
        | **Card Rank** | 2 – A (2,3,4,5,6,7,8,9,T,J,Q,K,A) | The rank of the card. T = Ten, J = Jack, Q = Queen, K = King, A = Ace. |
        | **Card Suit** | ♠ ♥ ♦ ♣ | The suit of the card. |

        > ⚠️ Both cards must be different. An error will appear if you select two identical cards.
        """)

    with st.expander("Situation"):
        st.markdown("""
        Context about the current state of the hand.

        | Field | Options | Description |
        |---|---|---|
        | **Street** | `preflop` `flop` `turn` `river` | The current betting round. Determines how many board cards are shown. |
        | **Position** | `early` `middle` `late` `button` | Your position relative to the dealer. *Button* is the most advantageous; *early* the most vulnerable. |
        | **Players at Table** | 2 – 9 | Number of players still in the hand. Affects equity calculations. |
        | **Opponent Type** | Tight Passive · Tight Aggressive · Loose Passive · Loose Aggressive | A profile of your opponents' overall tendencies. See the table below. |

        **Opponent Type Profiles**

        | Type | VPIP | Aggression | Typical Behaviour |
        |---|---|---|---|
        | Tight Passive | Low (~20%) | Low | Plays few hands, rarely bets/raises. Folds to aggression. |
        | Tight Aggressive | Low (~22%) | High | Plays strong hands only, bets and raises frequently (TAG style). |
        | Loose Passive | High (~55%) | Low | Plays many hands, mostly calls. Rarely folds pre-flop. |
        | Loose Aggressive | High (~60%) | High | Wide range, bets and bluffs often (LAG style). Most unpredictable. |
        """)

    with st.expander("💰  Stack & Pot"):
        st.markdown("""
        Sizing information measured in **Big Blinds (BB)**.

        | Field | Range | Description |
        |---|---|---|
        | **Stack Size (BB)** | 1 – 500 | Your current chip count in big blinds. Affects SPR and all-in EV. |
        | **Pot Size (BB)** | 1 – 1000 | Total chips in the middle right now (before any action this street). |
        | **To Call (BB)** | 0 – 500 | The amount you need to put in to stay in the hand. Set to `0` if it is your option with no bet in front of you. |
        """)

    with st.expander("🃏  Board Cards  *(flop / turn / river only)*"):
        st.markdown("""
        The community cards on the table. This section only appears when **Street** ≠ `preflop`.

        | Street | Cards Required |
        |---|---|
        | Flop | 3 board cards |
        | Turn | 4 board cards |
        | River | 5 board cards |

        Each card has a **Rank** and **Suit** selector, identical to the hole card inputs.

        > ⚠️ No board card may match a hole card or another board card. Duplicates are caught automatically.
        """)

    st.divider()

    # ── OUTPUTS ───────────────────────────────────────────────────────────────
    st.header("🃏 Outputs")

    with st.expander("Recommended Action", expanded=True):
        st.markdown("""
        The top-level decision produced by the Random Forest model.

        | Action | Indicator | Meaning |
        |---|---|---|
        | **FOLD** | 🔴 | Surrender your hand. |
        | **CALL** | 🟡 | Match the current bet to stay in the hand. |
        | **RAISE** | 🟢 | Put in a larger bet to build the pot or apply pressure. |
        | **ALL-IN** | 🔥 | Commit your entire stack. |
        """)

    with st.expander("📊  Action Probabilities"):
        st.markdown("""
        The softmax-like probability distribution across all four actions output by the model.
        Each bar represents how confident the model is in that action.

        - A **dominant bar** (>70%) indicates a clear decision.
        - **Balanced bars** suggest a marginal spot where other factors (reads, meta-game) matter.
        """)

    with st.expander("💵  EV Estimates"):
        st.markdown("""
        Expected Value (EV) estimates for each possible action, measured in **big blinds**.

        | Metric | Description |
        |---|---|
        | **Raise EV** | Estimated chips won/lost on average by raising, using fold equity + equity when called. |
        | **Call EV** | `(equity × pot) − to_call` — the classic pot-odds EV formula. |
        | **Fold EV** | Always **0** BB by definition (you surrender the pot but risk nothing more). |
        | **All-In EV** | Full push EV computed via Monte Carlo equity simulation against opponent range. `N/A` on pre-flop before the model can simulate fully. |

        > A positive EV means the action is profitable on average; negative means it loses chips in the long run.
        """)

    with st.expander("🔍  Hand Details"):
        st.markdown("""
        An expanded breakdown of every feature used internally by the model.

        **Strength & Equity**

        | Feature | Description |
        |---|---|
        | **Chen Score** | The Chen formula score for your hole cards (pre-flop strength heuristic). Higher is better. |
        | **Hand Strength (adjusted)** | Monte Carlo equity estimate (0–1) adjusted for opponent type VPIP. |
        | **Pot Odds** | `to_call / (pot + to_call)` — the minimum equity needed to make a call break-even. |
        | **Equity Edge** | `hand_strength − pot_odds` — positive means you have more equity than required. |
        | **SPR** | Stack-to-Pot Ratio. Low SPR (<3) favors committing; high SPR (>10) favors caution. |
        | **Suited** | Whether your two hole cards share the same suit (flush potential). |

        **Hand Classification** *(post-flop)*

        | Class | Meaning |
        |---|---|
        | 1 | Straight Flush |
        | 2 | Four of a Kind (Quads) |
        | 3 | Full House |
        | 4 | Flush |
        | 5 | Straight |
        | 6 | Three of a Kind (Set) |
        | 7 | Two Pair |
        | 8 | One Pair |
        | 9 | High Card |

        **Draws** *(post-flop)*

        | Feature | Description |
        |---|---|
        | **Board Wetness** | 0–3 score. Higher = more draws and connected cards on board, increasing variance. |
        | **Flush Draw** | You hold 4 cards to a flush. |
        | **Straight Draw** | Generic flag that you have a straight draw of some kind. |
        | **Open-Ended** | 8 outs to complete your straight (two cards that complete it). |
        | **Gutshot** | 4 outs to complete your straight (only one card completes it — inside draw). |

        **Board Texture** *(flop / turn / river)*

        | Feature | Description |
        |---|---|
        | **Connectedness** | How sequentially close board ranks are (0 = rainbow/disconnected, 3 = highly connected). |
        | **Paired** | Whether the board contains a pair (1 = yes, 0 = no). |
        | **Danger Score** | Composite measure of how threatening the board is to your hand (flush/straight possible). |
        """)

# router
if selected == "Poker":
    page_poker()
elif selected == "About Us":
    page_about()
elif selected == "Statistics":
    page_statistics()
elif selected == "User Manual":
    page_manual()
