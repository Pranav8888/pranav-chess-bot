import os
import shutil
from flask import Flask, render_template, request, jsonify
import chess
from main import SystemicAccumulatorBot, build_book_from_split_pgns

app = Flask(__name__)

# Dynamically locate Stockfish executable in Linux system PATH or environment variable
LOCAL_WINDOWS_PATH = r"C:\Users\nbala\Downloads\ChessBotX Trial\System\stockfish.exe"

if os.path.exists(LOCAL_WINDOWS_PATH):
    STOCKFISH_PATH = LOCAL_WINDOWS_PATH
else:
    STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", shutil.which("stockfish") or "/usr/games/stockfish")

WHITE_PGN_PATH = "white_repertoire.pgn"  
BLACK_PGN_PATH = "black_repertoire.pgn"

# Build the opening book on startup
opening_book = build_book_from_split_pgns(
    white_pgn_path=WHITE_PGN_PATH,
    black_pgn_path=BLACK_PGN_PATH,
    max_depth_moves=8
)

# Initialize bot with Elo rating and opening book
bot = SystemicAccumulatorBot(
    stockfish_path=STOCKFISH_PATH,
    elo=2050,
    opening_book=opening_book
)

board = chess.Board()
user_color = chess.WHITE  # Default color


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    global board, user_color
    data = request.json
    selected_color = data.get("color", "white")

    board = chess.Board()
    user_color = chess.WHITE if selected_color == "white" else chess.BLACK

    # If user chose Black, bot (White) makes the first move
    bot_move_uci = None
    if user_color == chess.BLACK:
        bot_move = bot.get_best_move(board, move_time=0.5)
        board.push(bot_move)
        bot_move_uci = bot_move.uci()

    return jsonify({
        "status": "started",
        "fen": board.fen(),
        "user_color": "white" if user_color == chess.WHITE else "black",
        "bot_move": bot_move_uci
    })


@app.route("/make_move", methods=["POST"])
def make_move():
    global board
    data = request.json
    user_move = data.get("move")

    # 1. Process User Move
    try:
        move = chess.Move.from_uci(user_move)
        if move in board.legal_moves:
            board.push(move)
        else:
            return jsonify({"status": "illegal", "fen": board.fen()})
    except Exception:
        return jsonify({"status": "invalid", "fen": board.fen()})

    if board.is_game_over():
        return jsonify({"status": "game_over", "result": board.result(), "fen": board.fen()})

    # 2. Get Bot Response
    bot_move = bot.get_best_move(board, move_time=0.5)
    board.push(bot_move)

    return jsonify({
        "status": "ok",
        "bot_move": bot_move.uci(),
        "fen": board.fen(),
        "game_over": board.is_game_over(),
        "result": board.result() if board.is_game_over() else None
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)