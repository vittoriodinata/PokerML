"""
Features.py  —  Shared feature engineering for PokerML
Imported by DataGen.py (data generation) and app.py (inference).
Keeping this in one place guarantees training and inference are always identical.
"""

from treys import Evaluator, Card

evaluator = Evaluator()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

POSITIONS      = ["early", "middle", "late", "button"]
OPPONENT_TYPES = ["tight_passive", "tight_aggressive", "loose_passive", "loose_aggressive"]

OPP_VPIP = {
    "tight_passive":    0.15,
    "tight_aggressive": 0.20,
    "loose_passive":    0.45,
    "loose_aggressive": 0.55,
}
OPP_AGGRESSION = {
    "tight_passive":    0.20,
    "tight_aggressive": 0.75,
    "loose_passive":    0.15,
    "loose_aggressive": 0.80,
}

# ── CARD HELPERS ──────────────────────────────────────────────────────────────

def card_rank(c: int) -> int:
    """Treys card int → rank as 2–14."""
    return Card.get_rank_int(c) + 2

def card_suit(c: int) -> int:
    """Treys card int → suit int (1, 2, 4, 8)."""
    return Card.get_suit_int(c)

# ── CHEN SCORE ────────────────────────────────────────────────────────────────

def chen_score_normalized(r1: int, r2: int, suited: int) -> float:
    """
    Chen formula normalised to [0, 1].
    r1, r2 are raw ranks (2–14); suited is 0/1.
    """
    high, low = max(r1, r2), min(r1, r2)
    score_map = {
        14: 10, 13: 8, 12: 7, 11: 6, 10: 5,
        9: 4.5, 8: 4, 7: 3.5, 6: 3, 5: 2.5,
        4: 2, 3: 1.5, 2: 1,
    }
    score = score_map.get(high, 1.0)

    if r1 == r2:
        score = max(score * 2, 5)

    if suited:
        score += 2

    gap = high - low if r1 != r2 else 0
    score -= {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}.get(min(gap, 4), 5)

    if 0 < gap <= 2:
        score += 1

    return round(max(0.0, min(1.0, score / 20.0)), 4)

# ── MULTIWAY EQUITY ───────────────────────────────────────────────────────────

def multiway_equity_factor(num_active: int) -> float:
    """
    Discount equity vs extra opponents.
    NOT applied to near-nut hands (hand_strength >= 0.95).
    """
    return 0.85 ** max(0, num_active - 2)

# ── HAND STRENGTH ─────────────────────────────────────────────────────────────

def get_hand_strength(board: list, hole: list, active_players: int) -> tuple:
    """
    Returns (hand_class, adjusted_hand_strength).
    Preflop  → Chen score x multiway factor.
    Postflop → treys evaluator; multiway factor skipped for near-nut hands (>= 0.95)
               so Royal Flush / Quads / etc. are never penalised.
    """
    if not board:
        r1, r2  = card_rank(hole[0]), card_rank(hole[1])
        suited  = int(card_suit(hole[0]) == card_suit(hole[1]))
        base    = chen_score_normalized(max(r1, r2), min(r1, r2), suited)
        return 9, round(base * multiway_equity_factor(active_players), 4)

    score         = evaluator.evaluate(board, hole)
    hand_class    = evaluator.get_rank_class(score)
    hand_strength = 1.0 - (score / 7462.0)

    # FIX #6: Near-nut hands are unbeatable — more opponents = bigger pot, not less equity.
    # This guard must exist in BOTH training (DataGen) and inference (app.py).
    if hand_strength >= 0.95:
        adjusted = hand_strength
    else:
        adjusted = hand_strength * multiway_equity_factor(active_players)

    return hand_class, round(adjusted, 4)

# ── HAND ABSTRACTION ─────────────────────────────────────────────────────────

def hand_abstraction_features(board: list, hole: list) -> dict:
    """
    Treys hand_class codes:
      1=Straight Flush, 2=Quads, 3=Full House, 4=Flush, 5=Straight,
      6=Set, 7=Two Pair, 8=One Pair, 9=High Card
    """
    if not board:
        return {
            "is_overpair": 0, "is_top_pair": 0, "is_middle_pair": 0,
            "is_set": 0, "is_two_pair": 0, "is_straight": 0,
            "is_flush": 0, "hand_abstraction": 0,
        }

    board_ranks = sorted([card_rank(c) for c in board])
    hole_ranks  = [card_rank(c) for c in hole]
    score       = evaluator.evaluate(board, hole)
    hand_class  = evaluator.get_rank_class(score)

    is_pair        = int(hand_class == 8)
    max_board      = max(board_ranks)
    mid_board      = sorted(board_ranks)[len(board_ranks) // 2]
    is_top_pair    = int(is_pair and max(hole_ranks) == max_board)
    is_middle_pair = int(is_pair and not is_top_pair and any(r == mid_board for r in hole_ranks))
    is_overpair    = int(hand_class == 8 and hole_ranks[0] == hole_ranks[1]
                         and min(hole_ranks) > max_board)
    is_set      = int(hand_class == 6)
    is_two_pair = int(hand_class == 7)
    is_straight = int(hand_class == 5)
    is_flush    = int(hand_class == 4)

    if hand_class <= 3:          # SF, Quads, Full House
        tier = 4
    elif hand_class <= 7:        # Flush, Straight, Set, Two Pair
        tier = 3
    elif is_overpair or is_top_pair:
        tier = 2
    elif is_pair:
        tier = 1
    else:
        tier = 0

    return {
        "is_overpair": is_overpair, "is_top_pair": is_top_pair,
        "is_middle_pair": is_middle_pair, "is_set": is_set,
        "is_two_pair": is_two_pair, "is_straight": is_straight,
        "is_flush": is_flush, "hand_abstraction": tier,
    }

# ── FLUSH FEATURES ────────────────────────────────────────────────────────────

def flush_features(hole: list, board: list) -> dict:
    if not board:
        return {"max_suit": 0, "flush_draw": 0, "flush_made": 0, "board_flush_pressure": 0}
    hole_suits  = [card_suit(c) for c in hole]
    board_suits = [card_suit(c) for c in board]
    all_suits   = hole_suits + board_suits
    player_max  = max(all_suits.count(s) for s in set(all_suits))
    board_max   = max(board_suits.count(s) for s in set(board_suits))
    return {
        "max_suit":             player_max,
        "flush_draw":           int(player_max == 4),
        "flush_made":           int(player_max >= 5),
        "board_flush_pressure": board_max,
    }

# ── STRAIGHT FEATURES ─────────────────────────────────────────────────────────

def straight_features(hole: list, board: list) -> dict:
    all_cards = hole + board
    ranks     = sorted(set(card_rank(c) for c in all_cards))
    if 14 in ranks:           # Ace can play low (wheel)
        ranks = [1] + ranks
    open_ended = gutshot = 0
    for low in range(1, 12):
        window = [r for r in ranks if low <= r <= low + 4]
        if len(window) >= 4:
            span = window[-1] - window[0]
            if span == 3:
                open_ended = 1          # OESD
            elif span == 4 and len(window) == 4:
                gutshot = 1             # GSD
    fd = flush_features(hole, board)["flush_draw"]
    if open_ended and fd:
        draw_type = 3
    elif open_ended:
        draw_type = 2
    elif gutshot:
        draw_type = 1
    else:
        draw_type = 0
    return {
        "straight_draw": int(open_ended or gutshot),
        "open_ended":    open_ended,
        "gutshot":       gutshot,
        "draw_type":     draw_type,
    }

# ── BOARD TEXTURE ─────────────────────────────────────────────────────────────

def board_wetness(board: list) -> int:
    """0=dry, 1=semi, 2=wet, 3=very wet."""
    if not board:
        return 0
    suits   = [card_suit(c) for c in board]
    board_r = sorted([card_rank(c) for c in board])
    flush_threat    = max(suits.count(s) for s in set(suits)) >= 3
    gaps            = [board_r[i + 1] - board_r[i] for i in range(len(board_r) - 1)]
    straight_threat = sum(1 for g in gaps if g <= 2) >= (len(board_r) - 1)
    is_paired       = len(board_r) != len(set(board_r))
    return min(3, int(flush_threat) + int(straight_threat) + int(is_paired))

def board_connectedness(board: list) -> int:
    if len(board) < 2:
        return 0
    ranks = sorted(set(card_rank(c) for c in board))
    max_run = current_run = 1
    for i in range(len(ranks) - 1):
        if ranks[i + 1] - ranks[i] == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run

def board_paired(board: list) -> int:
    ranks = [card_rank(c) for c in board]
    return int(len(ranks) != len(set(ranks)))

def board_high_card(board: list) -> int:
    return max(card_rank(c) for c in board) if board else 0

def board_danger_score(connectedness: int, flush_pressure: int, paired: int) -> int:
    return connectedness + flush_pressure + (2 * paired)

# ── POSITION HELPER ───────────────────────────────────────────────────────────

def seat_for_position(position: str, num_players: int) -> int:
    mapping = {
        "early":  1,
        "middle": max(1, num_players // 3),
        "late":   max(1, (2 * num_players) // 3),
        "button": num_players,
    }
    return mapping.get(position, 1)

# ── EV ────────────────────────────────────────────────────────────────────────

def calculate_ev(hand_strength: float, pot_odds: float,
                 to_call: float, pot_size: float, stack: float = None) -> dict:
    """
    Returns normalised EV for call, raise, fold, best, and optionally all-in.
    ev_allin is None when stack is not provided.

    FIX #2 / #8: `best_ev` and `ev_allin` keys are always present so both
    DataGen and app.py can unpack the dict without KeyError.
    """
    lose_prob = 1.0 - hand_strength
    norm      = max(pot_size, 1)
    ev_call   = hand_strength * (pot_size + to_call)   - lose_prob * to_call
    ev_raise  = hand_strength * (pot_size + 2*to_call) - lose_prob * 2*to_call
    ev_allin  = None
    if stack is not None:
        ev_allin = round((hand_strength * (pot_size + stack) - lose_prob * stack) / norm, 4)
    return {
        "ev_call":  round(ev_call  / norm, 4),
        "ev_raise": round(ev_raise / norm, 4),
        "ev_fold":  0.0,
        "best_ev":  round(max(ev_call, ev_raise, 0.0) / norm, 4),   # FIX #8
        "ev_allin": ev_allin,
    }

# ── ACTION LABELING (used by DataGen only) ────────────────────────────────────

def decide_action(hand_strength: float, pot_odds: float, spr: float,
                  position: str, active_players: int, opponent_type: str,
                  ev_dict: dict, wetness: int, stack: float) -> str:
    """Heuristic action label for dataset generation."""
    position_bonus   = {"button": 0.07, "late": 0.04, "middle": 0.0, "early": -0.05}
    pos_adj          = position_bonus.get(position, 0.0)
    multiway_penalty = 0.025 * max(0, active_players - 2)
    exploit_adj      = 0.03 if "tight" in opponent_type else -0.02

    adjusted         = hand_strength + pos_adj - multiway_penalty + exploit_adj
    edge             = adjusted - pot_odds
    raise_threshold  = 0.08 if spr <= 3 else 0.15

    if wetness >= 2 and 0.40 < hand_strength < 0.65:
        raise_threshold += 0.05

    # All-in conditions
    if stack < 20 and hand_strength > 0.45:
        return "all-in"
    if stack < 40 and hand_strength > 0.70:
        return "all-in"
    if spr < 2.0 and hand_strength > 0.60:
        return "all-in"
    if hand_strength > 0.90:
        return "all-in"
    if ev_dict["ev_raise"] > ev_dict["ev_call"] and spr < 4 and hand_strength > 0.75:
        return "all-in"

    # Standard actions
    if edge < -0.08:
        return "fold"
    elif ev_dict["ev_raise"] > ev_dict["ev_call"] and edge >= raise_threshold:
        return "raise"
    elif edge >= 0 or (edge >= -0.05 and ev_dict["ev_call"] > 0):
        return "call"
    else:
        return "fold"

# ── MAIN FEATURE BUILDER ──────────────────────────────────────────────────────

def build_feature_vector(hole: list, board: list, stage: str, position: str,
                         stack: float, pot: float, to_call: float,
                         players: int, opp_type: str,
                         street_bet_pressure: float = None) -> dict:
    """
    Given raw inputs (treys card ints + context), returns the full feature dict
    ready for the model. Used by both DataGen.py (training) and app.py (inference).

    Parameters
    ----------
    hole                 : list of 2 treys card ints
    board                : list of 0/3/4/5 treys card ints
    stage                : "preflop" | "flop" | "turn" | "river"
    position             : "early" | "middle" | "late" | "button"
    stack, pot, to_call  : float (big blinds)
    players              : int  (total players at table)
    opp_type             : one of OPPONENT_TYPES
    street_bet_pressure  : float 0-1; if None, derived from opp_aggression x 0.5
    """
    r1, r2 = card_rank(hole[0]), card_rank(hole[1])
    high, low = max(r1, r2), min(r1, r2)
    gap    = high - low
    # FIX #7: always use card_suit() (returns int) for both cards — never compare
    # the raw TREYS_SUIT string dict values, which gives a string == string result
    # that happens to work but is inconsistent with the card_suit() int path used
    # everywhere else (flush_features, board_wetness, etc.).
    suited = int(card_suit(hole[0]) == card_suit(hole[1]))

    hs_base              = chen_score_normalized(high, low, suited)
    hand_class, hand_str = get_hand_strength(board, hole, players)

    pot_odds       = to_call / max(pot + to_call, 1)
    spr            = stack   / max(pot + to_call, 1)
    seat_index     = seat_for_position(position, players)
    position_ratio = round(seat_index / max(players, 1), 4)
    effective_spr  = round(spr / max(players - 1, 1), 4)
    bet_ratio      = round(to_call / max(pot, 1), 4)
    pot_commitment = round(to_call / max(pot + to_call, 1), 4)
    equity_edge    = round(hand_str - pot_odds, 4)

    ev_dict   = calculate_ev(hand_str, pot_odds, to_call, pot, stack)
    abs_feats = hand_abstraction_features(board, hole)
    f_feats   = flush_features(hole, board)
    s_feats   = straight_features(hole, board)

    conn      = board_connectedness(board)
    paired    = board_paired(board)
    high_card = board_high_card(board)
    danger    = board_danger_score(conn, f_feats["board_flush_pressure"], paired)
    wetness   = board_wetness(board)

    if street_bet_pressure is None:
        street_bet_pressure = round(OPP_AGGRESSION[opp_type] * 0.5, 4)

    return {
        # Hole cards
        "rank1": high, "rank2": low, "suited": suited,
        "pair": int(r1 == r2), "gap": gap, "chen_score": hs_base,
        # Context
        "stage": stage, "position": position,
        "position_ratio": position_ratio, "seat_index": seat_index,
        "num_players": players, "active_players": players,
        # Stack / pot
        "stack_size": stack, "pot_size": pot, "to_call": to_call,
        "pot_odds": round(pot_odds, 4), "spr": round(spr, 4),
        "effective_spr": effective_spr, "bet_ratio": bet_ratio,
        "pot_commitment": pot_commitment,
        # EV
        "ev_call":  ev_dict["ev_call"],
        "ev_raise": ev_dict["ev_raise"],
        "ev_fold":  0.0,
        "best_ev":  ev_dict["best_ev"],
        # Equity
        "equity_edge": equity_edge,
        # Hand strength
        "hand_class": hand_class, "hand_strength": hand_str,
        # Abstraction
        **abs_feats,
        # Flush
        **f_feats,
        # Board texture
        "board_connectedness": conn, "board_paired": paired,
        "board_high_card": high_card, "board_danger_score": danger,
        "board_wetness": wetness,
        # Straight / draws
        **s_feats,
        # Opponent
        "opp_type_encoded":    OPPONENT_TYPES.index(opp_type),
        "opp_vpip":            OPP_VPIP[opp_type],
        "opp_aggression":      OPP_AGGRESSION[opp_type],
        "street_bet_pressure": street_bet_pressure,
        # ev_allin stored separately (not a model feature, used by UI only)
        "_ev_allin": ev_dict["ev_allin"],
    }