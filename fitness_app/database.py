"""
PostgreSQL database layer for FitAI (Supabase).
"""
import json
import os
import uuid
import bcrypt
import psycopg2
import psycopg2.extras
import psycopg2.errors
from datetime import date, datetime, timedelta
from contextlib import contextmanager


def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        database=os.environ.get("DB_NAME", "postgres"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
        sslmode="require",
    )


@contextmanager
def db():
    con = _connect()
    cur = con.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield cur
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        cur.close()
        con.close()


def init_db():
    with db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         TEXT UNIQUE,
                password_hash TEXT,
                name          TEXT NOT NULL,
                age           INTEGER,
                weight        REAL,
                height        REAL,
                goal          TEXT,
                fitness_level TEXT,
                is_premium    INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id       SERIAL PRIMARY KEY,
                user_id  INTEGER NOT NULL REFERENCES users(id),
                date     TEXT NOT NULL,
                notes    TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id           SERIAL PRIMARY KEY,
                session_id   INTEGER NOT NULL REFERENCES sessions(id),
                name         TEXT NOT NULL,
                sets         INTEGER,
                reps         INTEGER,
                weight       REAL,
                duration_min REAL,
                ex_type      TEXT DEFAULT 'strength'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS programs (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                name       TEXT NOT NULL,
                content    TEXT NOT NULL,
                is_ai      INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                expires_at TEXT NOT NULL
            )
        """)


# ── Auth ───────────────────────────────────────────────────────────────────────

def register_user(email, password, name, age, weight, height, goal, level):
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with db() as c:
            c.execute(
                "INSERT INTO users (email,password_hash,name,age,weight,height,goal,fitness_level) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (email.lower().strip(), pw_hash, name.strip(), age, weight, height, goal, level)
            )
            return c.fetchone()["id"]
    except psycopg2.errors.UniqueViolation:
        return None


def login_user(email, password):
    with db() as c:
        c.execute("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))
        row = c.fetchone()
    if row and row["password_hash"] and bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return dict(row)
    return None


def create_auth_token(user_id, days=30):
    token = str(uuid.uuid4())
    expires = (datetime.utcnow() + timedelta(days=days)).isoformat()
    with db() as c:
        c.execute("INSERT INTO auth_tokens (token,user_id,expires_at) VALUES (%s,%s,%s)",
                  (token, user_id, expires))
    return token


def validate_auth_token(token):
    with db() as c:
        c.execute(
            "SELECT u.* FROM auth_tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.token = %s AND t.expires_at > %s",
            (token, datetime.utcnow().isoformat())
        )
        row = c.fetchone()
    return dict(row) if row else None


def delete_auth_token(token):
    with db() as c:
        c.execute("DELETE FROM auth_tokens WHERE token = %s", (token,))


# ── Users ──────────────────────────────────────────────────────────────────────

def get_user(uid):
    with db() as c:
        c.execute("SELECT * FROM users WHERE id = %s", (uid,))
        row = c.fetchone()
    return dict(row) if row else None


def update_user(uid, **kwargs):
    fields = ", ".join(f"{k}=%s" for k in kwargs)
    with db() as c:
        c.execute(f"UPDATE users SET {fields} WHERE id=%s", (*kwargs.values(), uid))


# ── Sessions & Exercises ───────────────────────────────────────────────────────

def log_session(user_id, session_date, notes=""):
    with db() as c:
        c.execute(
            "INSERT INTO sessions (user_id,date,notes) VALUES (%s,%s,%s) RETURNING id",
            (user_id, str(session_date), notes)
        )
        return c.fetchone()["id"]


def log_exercise(session_id, name, sets=None, reps=None, weight=None, duration=None, ex_type="strength"):
    with db() as c:
        c.execute(
            "INSERT INTO exercises (session_id,name,sets,reps,weight,duration_min,ex_type) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (session_id, name, sets, reps, weight, duration, ex_type)
        )


def get_sessions(user_id, limit=30):
    with db() as c:
        c.execute("""
            SELECT s.*, COUNT(e.id) as ex_count
            FROM sessions s LEFT JOIN exercises e ON e.session_id = s.id
            WHERE s.user_id = %s GROUP BY s.id ORDER BY s.date DESC LIMIT %s
        """, (user_id, limit))
        return [dict(r) for r in c.fetchall()]


def get_session_exercises(session_id):
    with db() as c:
        c.execute("SELECT * FROM exercises WHERE session_id = %s", (session_id,))
        return [dict(r) for r in c.fetchall()]


# ── Programs ───────────────────────────────────────────────────────────────────

def save_program(user_id, name, content, is_ai=False):
    with db() as c:
        c.execute(
            "INSERT INTO programs (user_id,name,content,is_ai) VALUES (%s,%s,%s,%s) RETURNING id",
            (user_id, name, json.dumps(content), int(is_ai))
        )
        return c.fetchone()["id"]


def get_programs(user_id):
    with db() as c:
        c.execute("SELECT * FROM programs WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
        rows = c.fetchall()
    return [dict(r) | {"content": json.loads(r["content"])} for r in rows]


# ── Analytics ──────────────────────────────────────────────────────────────────

def get_streak(user_id):
    with db() as c:
        c.execute("SELECT DISTINCT date FROM sessions WHERE user_id=%s ORDER BY date DESC", (user_id,))
        rows = c.fetchall()
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
        c.execute("""
            SELECT e.*, s.date FROM exercises e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.user_id=%s AND LOWER(e.name)=LOWER(%s)
            ORDER BY s.date
        """, (user_id, exercise_name))
        return [dict(r) for r in c.fetchall()]
