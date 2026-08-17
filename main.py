"""
Systemic Accumulator chess bot — rewrite with PGN opening book parser.
"""

import chess
import chess.pgn
import time
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from stockfish import Stockfish


def build_book_from_split_pgns(
    white_pgn_path: str = None,
    black_pgn_path: str = None,
    max_depth_moves: int = 8,
) -> dict:
    """Combines separate White and Black PGN files into an opening book dictionary.

    :param white_pgn_path: Path to PGN file containing White openings.
    :param black_pgn_path: Path to PGN file containing Black openings.
    :param max_depth_moves: Max full moves (8 moves = 16 plies) to record.
    """
    position_counts = defaultdict(Counter)
    max_ply = max_depth_moves * 2

    def process_pgn(path: str, color: chess.Color):
        if not path:
            return
        with open(path, encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= max_ply:
                        break

                    # Store move if it matches target turn
                    if board.turn == color:
                        fields = board.fen().split(" ")
                        key = f"{fields[0]} {fields[1]}"
                        position_counts[key][move.uci()] += 1

                    board.push(move)

    process_pgn(white_pgn_path, chess.WHITE)
    process_pgn(black_pgn_path, chess.BLACK)

    # Map each position key to a list of tuples: [(uci_move, count), ...]
    opening_book = {}
    for key, counts in position_counts.items():
        # Store up to top 3 most common moves with their frequencies
        opening_book[key] = counts.most_common(3)

    return opening_book


@dataclass
class StyleConfig:
    base_depth: int = 8
    shallow_depth: int = 4
    conversion_depth: int = 10
    multipv: int = 5

    base_candidate_window_cp: float = 40.0
    window_volatility_scale: float = 0.15
    max_candidate_window_cp: float = 90.0

    conversion_threshold_cp: float = 120.0

    miss_base_prob: float = 0.05
    miss_complexity_scale_cp: float = 150.0
    miss_prob_cap: float = 0.80

    style_move_horizon: int = 30  # ply count over which opening-style bonuses decay to 0

    elo: int = 2050


class SystemicAccumulatorBot:
    def __init__(self, stockfish_path: str, cfg: StyleConfig = None,
                 opening_book: dict = None, elo: int = None):
        self.cfg = cfg or StyleConfig()
        if elo is not None:
            self.cfg.elo = elo

        self.book = opening_book or {}

        self.stockfish = Stockfish(
            path=stockfish_path,
            depth=self.cfg.base_depth,
            parameters={
                "Threads": 2,
                "Minimum Thinking Time": 0,
                "UCI_LimitStrength": True,
                "UCI_Elo": self.cfg.elo,
            },
        )
        self.stockfish.set_elo_rating(self.cfg.elo)

    # ------------------------------------------------------------------
    # Top-level move selection
    # ------------------------------------------------------------------

    def get_best_move(self, board: chess.Board, move_time: float = 0.5) -> chess.Move:
        start_time = time.time()
        fen = board.fen()

        book_move = self._book_lookup(board)
        if book_move is not None:
            self._apply_delay(start_time, move_time)
            return book_move

        complexity = self._measure_complexity(board)
        miss_prob = self._blunder_probability(complexity)

        self.stockfish.set_depth(self.cfg.base_depth)
        self.stockfish.set_fen_position(fen)
        top_moves = self.stockfish.get_top_moves(self.cfg.multipv)

        if not top_moves:
            chosen = self._fallback_move(board)
            self._apply_delay(start_time, move_time)
            return chosen

        best_cp = self._cp(top_moves[0])
        is_conversion = best_cp >= self.cfg.conversion_threshold_cp

        if is_conversion:
            self.stockfish.set_depth(self.cfg.conversion_depth)
            self.stockfish.set_fen_position(fen)
            top_moves = self.stockfish.get_top_moves(self.cfg.multipv)
            if top_moves:
                best_cp = self._cp(top_moves[0])

        window = self._candidate_window(complexity)
        candidates = [
            (chess.Move.from_uci(m["Move"]), self._cp(m))
            for m in top_moves
            if best_cp - self._cp(m) <= window
        ]
        if not candidates:
            candidates = [(chess.Move.from_uci(top_moves[0]["Move"]), best_cp)]

        if (not is_conversion) and random.random() < miss_prob:
            shallow_move = self._shallow_best_move(board)
            chosen = shallow_move if shallow_move is not None else candidates[0][0]
        elif is_conversion or len(candidates) == 1:
            chosen = candidates[0][0]
        else:
            chosen = self._pick_by_style(board, candidates)

        self.stockfish.set_depth(self.cfg.base_depth)
        self._apply_delay(start_time, move_time)
        return chosen

    def _fallback_move(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        return legal[0] if legal else None

    # ------------------------------------------------------------------
    # Complexity / miss modeling
    # ------------------------------------------------------------------

    def _measure_complexity(self, board: chess.Board) -> float:
        fen = board.fen()

        self.stockfish.set_depth(self.cfg.shallow_depth)
        self.stockfish.set_fen_position(fen)
        shallow = self.stockfish.get_top_moves(1)
        shallow_cp = self._cp(shallow[0]) if shallow else 0.0

        self.stockfish.set_depth(self.cfg.base_depth)
        self.stockfish.set_fen_position(fen)
        deep = self.stockfish.get_top_moves(1)
        deep_cp = self._cp(deep[0]) if deep else 0.0

        return abs(deep_cp - shallow_cp)

    def _blunder_probability(self, complexity_cp: float) -> float:
        if complexity_cp <= 0:
            return self.cfg.miss_base_prob
        scale = complexity_cp / self.cfg.miss_complexity_scale_cp
        prob = self.cfg.miss_base_prob + (1 - self.cfg.miss_base_prob) * (1 - math.exp(-scale))
        return min(prob, self.cfg.miss_prob_cap)

    def _shallow_best_move(self, board: chess.Board):
        self.stockfish.set_depth(self.cfg.shallow_depth)
        self.stockfish.set_fen_position(board.fen())
        top = self.stockfish.get_top_moves(1)
        self.stockfish.set_depth(self.cfg.base_depth)
        if not top:
            return None
        return chess.Move.from_uci(top[0]["Move"])

    # ------------------------------------------------------------------
    # Candidate window
    # ------------------------------------------------------------------

    def _candidate_window(self, complexity_cp: float) -> float:
        window = self.cfg.base_candidate_window_cp + complexity_cp * self.cfg.window_volatility_scale
        return min(window, self.cfg.max_candidate_window_cp)

    # ------------------------------------------------------------------
    # Style-based tie-breaking among near-equal candidates
    # ------------------------------------------------------------------

    def _pick_by_style(self, board: chess.Board, candidates):
        best_move = candidates[0][0]
        highest_score = -float("inf")
        for move, cp in candidates:
            score = cp + self._score_positional_style(board, move) + self._score_opponent_weaknesses(board, move)
            if score > highest_score:
                highest_score = score
                best_move = move
        return best_move

    def _score_positional_style(self, board: chess.Board, move: chess.Move) -> float:
        bonus = 0.0
        ply = board.ply()
        move_num = (ply // 2) + 1
        moving_piece = board.piece_at(move.from_square)

        if not moving_piece:
            return 0.0

        decay = max(0.0, 1.0 - (ply / self.cfg.style_move_horizon))

        if board.is_castling(move):
            return 30.0

        if decay > 0:
            if moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                from_rank = chess.square_rank(move.from_square)
                rel_from_rank = from_rank + 1 if board.turn == chess.WHITE else 8 - from_rank
                if rel_from_rank == 1:
                    bonus += 18.0 * decay

            if moving_piece.piece_type == chess.PAWN:
                to_sq = move.to_square
                to_file = chess.square_file(to_sq)
                to_rank = chess.square_rank(to_sq)
                rel_to_rank = to_rank + 1 if board.turn == chess.WHITE else 8 - to_rank

                if to_file in (2, 3, 4) and rel_to_rank in (3, 4):
                    bonus += 15.0 * decay

                board_after = board.copy()
                board_after.push(move)
                if self._defends_friendly_pawn(board_after, to_sq):
                    bonus += 12.0 * decay

                if to_file in (0, 1, 5, 6, 7):
                    if rel_to_rank > 3 and to_sq not in (chess.G3, chess.G6, chess.B3, chess.B6):
                        bonus -= 15.0 * decay

                if rel_to_rank > 4:
                    bonus -= 15.0 * decay

        if board.is_capture(move):
            target_piece = board.piece_at(move.to_square)
            if target_piece and moving_piece.piece_type == chess.PAWN and target_piece.piece_type == chess.PAWN:
                bonus -= 15.0 * max(decay, 0.3)
            if moving_piece.piece_type == chess.QUEEN and move_num <= 20:
                bonus -= 20.0

        return bonus

    def _score_opponent_weaknesses(self, board: chess.Board, move: chess.Move) -> float:
        bonus = 0.0
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece:
            return 0.0

        opponent = not moving_piece.color
        before_isolated = set(self._isolated_pawn_squares(board, opponent))

        if board.is_capture(move) and move.to_square in before_isolated:
            bonus += 15.0

        board_after = board.copy()
        board_after.push(move)

        if moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if self._is_outpost_square(board_after, move.to_square, moving_piece.color):
                bonus += 20.0

        after_isolated = set(self._isolated_pawn_squares(board_after, opponent))
        newly_isolated = after_isolated - before_isolated
        if newly_isolated:
            bonus += 10.0 * len(newly_isolated)

        return bonus

    # ------------------------------------------------------------------
    # Structural helpers
    # ------------------------------------------------------------------

    def _defends_friendly_pawn(self, board: chess.Board, square: chess.Square) -> bool:
        moved_color = not board.turn
        for atk_sq in board.attacks(square):
            piece = board.piece_at(atk_sq)
            if piece and piece.piece_type == chess.PAWN and piece.color == moved_color:
                return True
        return False

    def _file_has_pawn(self, board: chess.Board, file_idx: int, color: bool) -> bool:
        for sq in chess.SQUARES:
            if chess.square_file(sq) == file_idx:
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    return True
        return False

    def _isolated_pawn_squares(self, board: chess.Board, color: bool):
        isolated = []
        for sq in board.pieces(chess.PAWN, color):
            f = chess.square_file(sq)
            left = f - 1 >= 0 and self._file_has_pawn(board, f - 1, color)
            right = f + 1 <= 7 and self._file_has_pawn(board, f + 1, color)
            if not left and not right:
                isolated.append(sq)
        return isolated

    def _is_outpost_square(self, board: chess.Board, square: chess.Square, color: bool) -> bool:
        f = chess.square_file(square)
        r = chess.square_rank(square)
        rel_rank = r + 1 if color == chess.WHITE else 8 - r
        if rel_rank < 4:
            return False
        enemy = not color
        for adj_f in (f - 1, f + 1):
            if 0 <= adj_f <= 7 and self._file_has_pawn(board, adj_f, enemy):
                return False
        return True

    # ------------------------------------------------------------------
    # Opening book
    # ------------------------------------------------------------------

    def _book_lookup(self, board: chess.Board):
        if not self.book:
            return None
        fields = board.fen().split(" ")
        key = f"{fields[0]} {fields[1]}"
        entries = self.book.get(key)
        if not entries:
            return None

        # Format normalizer: converts legacy single string or simple lists to tuples
        normalized_entries = []
        if isinstance(entries, str):
            entries = [(entries, 1)]
        for item in entries:
            if isinstance(item, str):
                normalized_entries.append((item, 1))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                normalized_entries.append(item)

        candidates = []
        weights = []

        for uci, count in normalized_entries:
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    candidates.append(move)
                    weights.append(count)
            except ValueError:
                continue

        if not candidates:
            return None

        # Weighted random selection based on historical occurrence count
        return random.choices(candidates, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _cp(self, entry) -> float:
        cp = entry["Centipawn"]
        if cp is None:
            cp = 10000 if (entry["Mate"] or 0) > 0 else -10000
        return float(cp)

    def _apply_delay(self, start_time: float, target_delay: float = 1.0):
        elapsed = time.time() - start_time
        if elapsed < target_delay:
            time.sleep(target_delay - elapsed)


if __name__ == "__main__":
    STOCKFISH_PATH = r"C:\Users\nbala\Downloads\ChessBotX Trial\System\stockfish.exe"
    
    # Example loading from PGNs
    opening_book = build_book_from_split_pgns(
        white_pgn_path="white_repertoire.pgn",
        black_pgn_path="black_repertoire.pgn",
        max_depth_moves=12
    )

    bot = SystemicAccumulatorBot(STOCKFISH_PATH, elo=2050, opening_book=opening_book)
    board = chess.Board()
    move = bot.get_best_move(board, move_time=0.5)
    print(board.san(move))