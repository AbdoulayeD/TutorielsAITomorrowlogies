"""
SQLite database layer for FitAI.
All CRUD operations for users, sessions, exercises, and programs.
"""
import sqlite3
import json
import uuid
import bcrypt
from datetime import date, datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "fitness.db"


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    UNIQUE,
            password_hash TEXT,
            name          TEXT    NOT NULL,
            age           INTEGER,
            weight        REAL,
            height        REAL,
            goal          TEXT,
            fitness_level TEXT,
            is_premium    INTEGER DEFAULT 0,
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            date        TEXT    NOT NULL,
            notes       TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS exercises (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   INTEGER NOT NULL,
            name         TEXT    NOT NULL,
            sets         INTEGER,
            reps         INTEGER,
            weight       REAL,
            duration_min REAL,
            ex_type      TEXT    DEFAULT 'strength',
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS programs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            name        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            is_ai       INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS auth_tokens (
            token       TEXT    PRIMARY KEY,
            user_id     INTEGER NOT NULL,
            expires_at  TEXT    NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)
        # Migrate existing databases that lack the new auth columns
        existing = {r[1] for r in c.execute("PRAGMA table_info(users)").fetchall()}
        if "email" not in existing:
            c.execute("ALTER TABLE users ADD COLUMN email TEXT")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL")
        if "password_hash" not in existing:
            c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")


# ── Auth ───────────────────────────────────────────────────────────────────────

def register_user(email: str, password: str, name: str, age: int, weight: float,
                  height: float, goal: str, level: str) -> int | None:
    """Create a new account. Returns user id, or None if email already exists."""
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with db() as c:
            cur = c.execute(
                "INSERT INTO users (email,password_hash,name,age,weight,height,goal,fitness_level) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (email.lower().strip(), pw_hash, name.strip(), age, weight, height, goal, level)
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def login_user(email: str, password: str) -> dict | None:
    """Verify credentials. Returns user dict on success, None on failure."""
    with db() as c:
        row = c.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
    if row and row["password_hash"] and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return dict(row)
    return None


def create_auth_token(user_id: int, days: int = 30) -> str:
    token = str(uuid.uuid4())
    expires = (datetime.utcnow() + timedelta(days=days)).isoformat()
    with db() as c:
        c.execute("INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?,?,?)",
                  (token, user_id, expires))
    return token


def validate_auth_token(token: str) -> dict | None:
    """Returns user dict if token is valid and not expired, else None."""
    with db() as c:
        row = c.execute(
            "SELECT u.* FROM auth_tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token = ? AND t.expires_at > ?",
            (token, datetime.utcnow().isoformat())
        ).fetchone()
    return dict(row) if row else None


def delete_auth_token(token: str):
    with db() as c:
        c.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


# ── Users ──────────────────────────────────────────────────────────────────────

def get_users():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()]


def get_user(uid: int):
    with db() as c:
        r = c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return dict(r) if r else None


def create_user(name, age, weight, height, goal, level):
    with db() as c:
        cur = c.execute(
            "INSERT INTO users (name,age,weight,height,goal,fitness_level) VALUES (?,?,?,?,?,?)",
            (name, age, weight, height, goal, level)
        )
        return cur.lastrowid


def update_user(uid: int, **kwargs):
    fields = ", ".join(f"{k}=?" for k in kwargs)
    with db() as c:
        c.execute(f"UPDATE users SET {fields} WHERE id=?", (*kwargs.values(), uid))


# ── Sessions & Exercises ───────────────────────────────────────────────────────

def log_session(user_id, session_date, notes=""):
    with db() as c:
        cur = c.execute(
            "INSERT INTO sessions (user_id,date,notes) VALUES (?,?,?)",
            (user_id, str(session_date), notes)
        )
        return cur.lastrowid


def log_exercise(session_id, name, sets=None, reps=None, weight=None, duration=None, ex_type="strength"):
    with db() as c:
        c.execute(
            "INSERT INTO exercises (session_id,name,sets,reps,weight,duration_min,ex_type) VALUES (?,?,?,?,?,?,?)",
            (session_id, name, sets, reps, weight, duration, ex_type)
        )


def get_sessions(user_id, limit=30):
    with db() as c:
        rows = c.execute("""
            SELECT s.*, COUNT(e.id) as ex_count
            FROM sessions s LEFT JOIN exercises e ON e.session_id = s.id
            WHERE s.user_id = ? GROUP BY s.id ORDER BY s.date DESC LIMIT ?
        """, (user_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_session_exercises(session_id):
    with db() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM exercises WHERE session_id = ?", (session_id,)
        ).fetchall()]


# ── Programs ───────────────────────────────────────────────────────────────────

def save_program(user_id, name, content, is_ai=False):
    with db() as c:
        cur = c.execute(
            "INSERT INTO programs (user_id,name,content,is_ai) VALUES (?,?,?,?)",
            (user_id, name, json.dumps(content), int(is_ai))
        )
        return cur.lastrowid


def get_programs(user_id):
    with db() as c:
        rows = c.execute(
            "SELECT * FROM programs WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) | {"content": json.loads(r["content"])} for r in rows]


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_streak(user_id: int) -> int:
    with db() as c:
        rows = c.execute(
            "SELECT DISTINCT date FROM sessions WHERE user_id=? ORDER BY date DESC", (user_id,)
        ).fetchall()
    if not rows:
        return 0
    streak = 0
    today = date.today()
    for i, row in enumerate(rows):
        diff = (today - date.fromisoformat(row["date"])).days
        if diff == i or diff == i + 1:
            streak += 1
        else:
            break
    return streak


def get_exercise_history(user_id, exercise_name):
    with db() as c:
        return [dict(r) for r in c.execute("""
            SELECT e.*, s.date FROM exercises e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.user_id=? AND LOWER(e.name)=LOWER(?)
            ORDER BY s.date
        """, (user_id, exercise_name)).fetchall()]
