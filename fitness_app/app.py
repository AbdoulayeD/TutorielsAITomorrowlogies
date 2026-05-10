"""
FitAI — AI-powered fitness tracker
===================================
Free features  : workout logging, history, static templates, streak
Premium (💎)   : AI program generation, dynamic adaptation, progress charts, AI Coach chat

Stack : Streamlit + SQLite + Anthropic Claude
Deploy: streamlit run app.py  →  https://streamlit.io/cloud (free, web + mobile browser)
"""

import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

# On Streamlit Cloud secrets come from st.secrets; locally from .env
try:
    for key in ("GEMINI_API_KEY", "DATABASE_URL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass
import pandas as pd
from datetime import date

from database import (
    init_db, get_user, update_user,
    log_session, get_sessions, log_exercise, get_session_exercises,
    get_exercise_history, save_program, get_programs, get_streak,
    register_user, login_user, create_auth_token, validate_auth_token, delete_auth_token,
)

# ── Config ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FitAI 💪",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Mobile-friendly tweaks + dark card style
st.markdown("""
<style>
  .block-container { padding-top: 0.75rem; padding-bottom: 4rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 0; }
  .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 0.45rem 0.6rem; }
  div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 0.6rem; }
  @media (max-width: 480px) {
    .stTabs [data-baseweb="tab"] { font-size: 0.68rem; padding: 0.35rem 0.4rem; }
  }
</style>
""", unsafe_allow_html=True)

init_db()

# ── Session state defaults ──────────────────────────────────────────────────────

if "uid" not in st.session_state:
    st.session_state.uid = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ── Restore session from URL query param ───────────────────────────────────────

if not st.session_state.uid:
    token = st.query_params.get("session")
    if token:
        user_from_token = validate_auth_token(token)
        if user_from_token:
            st.session_state.uid = user_from_token["id"]

# ── Auth gate ──────────────────────────────────────────────────────────────────

if not st.session_state.uid:
    st.title("💪 FitAI")
    st.markdown("Your AI-powered fitness companion.")
    st.divider()

    tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In", type="primary", use_container_width=True):
                user = login_user(email, password)
                if user:
                    token = create_auth_token(user["id"])
                    st.query_params["session"] = token
                    st.session_state.uid = user["id"]
                    st.session_state.chat = []
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with tab_register:
        with st.form("register_form"):
            name   = st.text_input("Full name")
            email  = st.text_input("Email", key="reg_email")
            pw     = st.text_input("Password", type="password", key="reg_pw")
            pw2    = st.text_input("Confirm password", type="password")
            c1, c2 = st.columns(2)
            age    = c1.number_input("Age",         10,  100,  25)
            weight = c2.number_input("Weight (kg)", 30.0, 300.0, 70.0, step=0.5)
            c3, c4 = st.columns(2)
            height = c3.number_input("Height (cm)", 100, 250, 170)
            level  = c4.selectbox("Fitness level", ["Beginner", "Intermediate", "Advanced"])
            goal   = st.selectbox("Primary goal", [
                "Build muscle", "Lose weight / Fat loss", "Improve endurance",
                "Increase strength", "General fitness", "Athletic performance",
                "Body recomposition",
            ])
            if st.form_submit_button("🚀 Create Account", type="primary", use_container_width=True):
                if not name.strip() or not email.strip() or not pw:
                    st.error("Please fill in all fields.")
                elif pw != pw2:
                    st.error("Passwords do not match.")
                elif len(pw) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    uid = register_user(email, pw, name, age, weight, height, goal, level)
                    if uid:
                        token = create_auth_token(uid)
                        st.query_params["session"] = token
                        st.session_state.uid = uid
                        st.session_state.chat = []
                        st.rerun()
                    else:
                        st.error("An account with this email already exists.")
    st.stop()

# ── Logged-in header ───────────────────────────────────────────────────────────

user = get_user(st.session_state.uid)

with st.sidebar:
    st.markdown(f"**{user['name']}**")
    st.caption(user.get("email") or "")
    if st.button("🚪 Log out", use_container_width=True):
        token = st.query_params.get("session")
        if token:
            delete_auth_token(token)
        st.query_params.clear()
        st.session_state.uid = None
        st.session_state.chat = []
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_home, tab_log, tab_prog, tab_stats, tab_ai = st.tabs(
    ["🏠 Home", "📝 Log", "📋 Programs", "📊 Progress", "🤖 AI Coach"]
)

# ═══════════════════════════════════════════════════════
# TAB 1 — HOME / DASHBOARD
# ═══════════════════════════════════════════════════════

with tab_home:
    streak       = get_streak(st.session_state.uid)
    all_sessions = get_sessions(st.session_state.uid, limit=100)
    today        = str(date.today())
    today_done   = any(s["date"] == today for s in all_sessions[:7])

    st.markdown(f"### Hey {user['name']}! 👋")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔥 Streak",   f"{streak} d")
    c2.metric("💪 Workouts", len(all_sessions))
    c3.metric("🎯 Goal",     user["goal"].split("/")[0].strip()[:12])
    c4.metric("⚡ Level",    user["fitness_level"][:5])

    st.markdown("#### Today")
    if today_done:
        st.success("✅ Workout logged today!")
    else:
        st.warning("🏋️ No workout logged yet today.")

    st.markdown("#### Recent Workouts")
    recent = all_sessions[:5]
    if recent:
        for s in recent:
            d        = date.fromisoformat(s["date"])
            days_ago = (date.today() - d).days
            label    = "Today" if days_ago == 0 else ("Yesterday" if days_ago == 1 else f"{days_ago}d ago")
            with st.container(border=True):
                cc1, cc2 = st.columns([3, 1])
                cc1.markdown(f"**{label}** — {s['ex_count']} exercises")
                cc2.caption(s["date"])
                if s.get("notes"):
                    st.caption(s["notes"])
    else:
        st.info("No workouts yet — start logging to build your history!")

    # ── Freemium upgrade prompt ──
    if not user["is_premium"]:
        st.divider()
        with st.container(border=True):
            st.markdown("#### 💎 Upgrade to Premium")
            st.markdown(
                "Unlock **AI-generated programs**, dynamic adaptation after missed sessions, "
                "exercise progression charts, and a personal AI Coach chat."
            )
            col_free, col_paid = st.columns(2)
            with col_free:
                st.markdown("**🆓 Free**")
                st.markdown("- Workout logging\n- Exercise tracker\n- Streak\n- Static templates\n- History")
            with col_paid:
                st.markdown("**💎 Premium**")
                st.markdown("- AI program generation\n- Session adaptation\n- Progress charts\n- AI Coach chat\n- Nutrition tips")
            if st.button("⬆️ Activate Premium", type="primary", use_container_width=True):
                update_user(st.session_state.uid, is_premium=1)
                st.success("🎉 Premium activated!")
                st.rerun()
    else:
        st.success("💎 Premium active — all features unlocked!")

# ═══════════════════════════════════════════════════════
# TAB 2 — LOG WORKOUT
# ═══════════════════════════════════════════════════════

with tab_log:
    st.markdown("### 📝 Log Workout")

    with st.form("workout_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        session_date = c1.date_input("Date", date.today())
        notes        = c2.text_input("Notes", placeholder="PR day, felt strong…")

        n_ex = st.slider("Number of exercises", 1, 12, 4)
        exercises_data = []

        for i in range(n_ex):
            st.markdown(f"**Exercise {i + 1}**")
            cn, ct = st.columns([3, 1])
            ex_name = cn.text_input("Name", key=f"en{i}", placeholder="e.g. Squat, Running…")
            ex_type = ct.selectbox("Type", ["Strength", "Cardio"], key=f"et{i}")

            if ex_type == "Strength":
                cs, cr, cw = st.columns(3)
                sets   = cs.number_input("Sets",   1, 20,    3,   key=f"es{i}")
                reps   = cr.number_input("Reps",   1, 100,  10,   key=f"er{i}")
                weight = cw.number_input("kg",     0.0, 500.0, 0.0, key=f"ew{i}", step=2.5)
                exercises_data.append({"name": ex_name, "type": "strength",
                                       "sets": sets, "reps": reps, "weight": weight})
            else:
                cd, ci = st.columns(2)
                duration  = cd.number_input("Minutes", 1, 300, 30, key=f"ed{i}")
                intensity = ci.selectbox("Intensity", ["Low", "Moderate", "High", "HIIT"], key=f"ei{i}")
                exercises_data.append({"name": ex_name, "type": "cardio",
                                       "duration": duration, "intensity": intensity})

            if i < n_ex - 1:
                st.markdown("---")

        submitted = st.form_submit_button("💾 Save Workout", type="primary", use_container_width=True)

    if submitted:
        sid    = log_session(st.session_state.uid, session_date, notes)
        logged = 0
        for ex in exercises_data:
            if ex["name"].strip():
                if ex["type"] == "strength":
                    log_exercise(sid, ex["name"], ex["sets"], ex["reps"], ex["weight"], ex_type="strength")
                else:
                    log_exercise(sid, ex["name"], duration=ex["duration"], ex_type="cardio")
                logged += 1
        if logged > 0:
            st.success(f"✅ {logged} exercise(s) saved for {session_date}!")
            st.balloons()
        else:
            st.warning("No exercises saved — fill in at least one exercise name.")

# ═══════════════════════════════════════════════════════
# TAB 3 — PROGRAMS
# ═══════════════════════════════════════════════════════

FREE_TEMPLATES = {
    "StrongLifts 5×5  (3-day, beginner)": {
        "description": "The most proven beginner strength program. Add 2.5 kg every session.",
        "days_per_week": 3,
        "days": [
            {"label": "Workout A — Mon/Fri", "exercises": [
                {"name": "Squat",        "sets": 5, "reps": "5"},
                {"name": "Bench Press",  "sets": 5, "reps": "5"},
                {"name": "Barbell Row",  "sets": 5, "reps": "5"},
            ]},
            {"label": "Workout B — Wed", "exercises": [
                {"name": "Squat",           "sets": 5, "reps": "5"},
                {"name": "Overhead Press",  "sets": 5, "reps": "5"},
                {"name": "Deadlift",        "sets": 1, "reps": "5"},
            ]},
        ],
    },
    "Push / Pull / Legs  (6-day, intermediate)": {
        "description": "High-frequency 6-day split for hypertrophy. Alternate A/B weeks.",
        "days_per_week": 6,
        "days": [
            {"label": "Monday — Push (Chest · Shoulders · Triceps)", "exercises": [
                {"name": "Bench Press",       "sets": 4, "reps": "6–10"},
                {"name": "Overhead Press",    "sets": 3, "reps": "8–12"},
                {"name": "Incline DB Press",  "sets": 3, "reps": "10–12"},
                {"name": "Lateral Raises",    "sets": 4, "reps": "15–20"},
                {"name": "Tricep Pushdown",   "sets": 3, "reps": "12–15"},
            ]},
            {"label": "Tuesday — Pull (Back · Biceps)", "exercises": [
                {"name": "Pull-ups",          "sets": 4, "reps": "6–10"},
                {"name": "Barbell Row",       "sets": 4, "reps": "8–10"},
                {"name": "Face Pull",         "sets": 3, "reps": "15–20"},
                {"name": "Barbell Curl",      "sets": 3, "reps": "10–12"},
            ]},
            {"label": "Wednesday — Legs", "exercises": [
                {"name": "Squat",                 "sets": 4, "reps": "6–10"},
                {"name": "Romanian Deadlift",     "sets": 3, "reps": "8–12"},
                {"name": "Leg Press",             "sets": 3, "reps": "12–15"},
                {"name": "Calf Raises",           "sets": 4, "reps": "15–20"},
            ]},
            {"label": "Thu–Sat: repeat Push/Pull/Legs", "exercises": []},
        ],
    },
    "Fat Loss  (4-day, any level)": {
        "description": "2 strength + 2 cardio days. Maximises caloric burn while preserving muscle.",
        "days_per_week": 4,
        "days": [
            {"label": "Monday — Upper Strength", "exercises": [
                {"name": "DB Bench Press",    "sets": 3, "reps": "12–15"},
                {"name": "Cable Row",         "sets": 3, "reps": "12–15"},
                {"name": "Overhead Press",    "sets": 3, "reps": "12–15"},
                {"name": "Lateral Raises",    "sets": 3, "reps": "15–20"},
            ]},
            {"label": "Tuesday — LISS Cardio", "exercises": [
                {"name": "Walk / Bike / Elliptical", "duration_min": 45, "type": "cardio"},
            ]},
            {"label": "Thursday — Lower Strength", "exercises": [
                {"name": "Goblet Squat",      "sets": 3, "reps": "15"},
                {"name": "Hip Thrust",        "sets": 3, "reps": "12–15"},
                {"name": "Walking Lunges",    "sets": 3, "reps": "12 each"},
                {"name": "Leg Curl",          "sets": 3, "reps": "12–15"},
            ]},
            {"label": "Saturday — HIIT", "exercises": [
                {"name": "HIIT Circuit (bike / sprints / burpees)", "duration_min": 25, "type": "cardio"},
            ]},
        ],
    },
    "Endurance Base  (5-day, beginner/intermediate)": {
        "description": "Build your aerobic base with structured run/bike sessions + 2 strength days.",
        "days_per_week": 5,
        "days": [
            {"label": "Monday — Easy Run", "exercises": [
                {"name": "Easy jog / zone 2", "duration_min": 35, "type": "cardio"},
            ]},
            {"label": "Tuesday — Strength (Full Body)", "exercises": [
                {"name": "Squat",        "sets": 3, "reps": "10"},
                {"name": "Pull-ups",     "sets": 3, "reps": "8"},
                {"name": "Push-ups",     "sets": 3, "reps": "15"},
                {"name": "Plank",        "sets": 3, "reps": "45s"},
            ]},
            {"label": "Wednesday — Tempo Run", "exercises": [
                {"name": "Tempo run (zone 3–4)", "duration_min": 30, "type": "cardio"},
            ]},
            {"label": "Friday — Strength (Full Body)", "exercises": [
                {"name": "Deadlift",     "sets": 3, "reps": "8"},
                {"name": "Dumbbell Row", "sets": 3, "reps": "10"},
                {"name": "Lunges",       "sets": 3, "reps": "12"},
            ]},
            {"label": "Sunday — Long Easy Run", "exercises": [
                {"name": "Long run / bike (zone 2)", "duration_min": 60, "type": "cardio"},
            ]},
        ],
    },
}

with tab_prog:
    st.markdown("### 📋 Programs")
    pt_free, pt_mine = st.tabs(["🆓 Free Templates", "💾 My Programs"])

    with pt_free:
        tmpl_name = st.selectbox("Choose template", list(FREE_TEMPLATES.keys()))
        tmpl      = FREE_TEMPLATES[tmpl_name]
        st.caption(f"📅 {tmpl['days_per_week']} days/week — {tmpl['description']}")

        for day in tmpl["days"]:
            with st.expander(day["label"]):
                exs = day["exercises"]
                if not exs:
                    st.caption("(same pattern — see above days)")
                for ex in exs:
                    if "duration_min" in ex:
                        st.write(f"• **{ex['name']}** — {ex['duration_min']} min cardio")
                    else:
                        st.write(f"• **{ex['name']}** — {ex['sets']} × {ex['reps']}")

        if st.button("💾 Save Template to My Programs", use_container_width=True):
            save_program(st.session_state.uid, tmpl_name, tmpl, is_ai=False)
            st.success(f'"{tmpl_name}" saved to your programs!')

    with pt_mine:
        my_programs = get_programs(st.session_state.uid)
        if not my_programs:
            st.info("No programs saved yet. Save a template above, or generate an AI program (Premium 💎).")
        else:
            for prog in my_programs:
                badge   = "🤖 AI" if prog["is_ai"] else "📋"
                content = prog["content"]
                with st.expander(f"{badge} {prog['name']}  ·  {prog['created_at'][:10]}"):
                    if content.get("description"):
                        st.caption(content["description"])
                    days = content.get("days", content.get("schedule", []))
                    for day in days[:5]:
                        label = day.get("label", day.get("day", "Day"))
                        st.markdown(f"**{label}**")
                        for ex in day.get("exercises", [])[:6]:
                            if "duration_min" in ex:
                                st.write(f"  • {ex['name']}: {ex.get('duration_min')} min")
                            else:
                                s = ex.get("sets", "-")
                                r = ex.get("reps", "-")
                                st.write(f"  • {ex['name']}: {s} × {r}")
                    # AI programs have a 'weeks' structure
                    for week in content.get("weeks", [])[:1]:
                        st.markdown(f"*Week {week['week']}* — {week.get('notes', '')}")
                        days = week.get("days", [])
                        if isinstance(days, str):
                            st.caption(days)
                        else:
                            for day in days:
                                st.markdown(f"**{day.get('day', '')}** — {day.get('focus', '')}")
                                for ex in day.get("exercises", [])[:5]:
                                    if "duration_min" in ex:
                                        st.write(f"  • {ex['name']}: {ex.get('duration_min')} min")
                                    else:
                                        st.write(f"  • {ex['name']}: {ex.get('sets')} × {ex.get('reps')}")

# ═══════════════════════════════════════════════════════
# TAB 4 — PROGRESS
# ═══════════════════════════════════════════════════════

with tab_stats:
    st.markdown("### 📊 Progress")

    all_sessions = get_sessions(st.session_state.uid, limit=100)

    if not all_sessions:
        st.info("Log some workouts to see your progress here!")
    else:
        # ── Free: history table ──
        st.subheader("Workout History")
        df = pd.DataFrame(all_sessions)
        df["date"] = pd.to_datetime(df["date"])
        st.dataframe(
            df[["date", "ex_count", "notes"]].rename(
                columns={"date": "Date", "ex_count": "Exercises", "notes": "Notes"}
            ),
            hide_index=True, use_container_width=True,
        )

        if user["is_premium"]:
            import plotly.express as px

            st.divider()
            st.subheader("💎 Exercise Progression")

            # Collect unique strength exercise names from all sessions
            all_ex_names = []
            for s in all_sessions:
                all_ex_names.extend(
                    e["name"] for e in get_session_exercises(s["id"]) if e["ex_type"] == "strength"
                )
            unique_ex = sorted(set(all_ex_names))

            if unique_ex:
                sel_ex  = st.selectbox("Track exercise", unique_ex)
                history = get_exercise_history(st.session_state.uid, sel_ex)

                if len(history) > 1:
                    df_h           = pd.DataFrame(history)
                    df_h["date"]   = pd.to_datetime(df_h["date"])
                    df_h           = df_h.sort_values("date")
                    df_h["volume"] = df_h["sets"] * df_h["reps"] * df_h["weight"]

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.line(df_h, x="date", y="weight", title=f"{sel_ex} — Weight (kg)", markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig2 = px.bar(df_h, x="date", y="volume", title=f"{sel_ex} — Volume (sets×reps×kg)")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("Log this exercise at least twice to see a progression chart.")

            st.divider()
            st.subheader("Weekly Session Frequency")
            df_freq          = pd.DataFrame(all_sessions)
            df_freq["date"]  = pd.to_datetime(df_freq["date"])
            df_freq["isoyr"] = df_freq["date"].dt.isocalendar().year.astype(str)
            df_freq["isoWk"] = df_freq["date"].dt.isocalendar().week.astype(str).str.zfill(2)
            df_freq["week"]  = df_freq["isoyr"] + "-W" + df_freq["isoWk"]
            weekly = df_freq.groupby("week").size().reset_index(name="sessions")
            fig3   = px.bar(weekly.tail(16), x="week", y="sessions", title="Sessions per Week (last 16 weeks)")
            st.plotly_chart(fig3, use_container_width=True)

        else:
            st.divider()
            st.info("💎 **Premium**: exercise progression charts, volume tracking, and weekly analysis.")

# ═══════════════════════════════════════════════════════
# TAB 5 — AI COACH  (Premium only)
# ═══════════════════════════════════════════════════════

with tab_ai:
    st.markdown("### 🤖 AI Coach")

    if not user["is_premium"]:
        st.warning("💎 **Premium feature** — upgrade to unlock AI coaching.")
        st.markdown("""
**What you get with Premium AI Coach:**
- 🎯 **Personalized program** built on your goals + latest sports science
- 🔄 **Dynamic adaptation** when you miss a session — AI reshuffles intelligently
- 💬 **Coach chat** — training, nutrition, recovery, form questions answered
- 📈 **Progressive overload guidance** — when and how much to add
- 🍽️ **Nutrition tips** aligned with your goal
        """)
        if st.button("⬆️ Activate Premium Now", type="primary", use_container_width=True):
            update_user(st.session_state.uid, is_premium=1)
            st.success("🎉 Premium activated!")
            st.rerun()
        st.stop()

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Add it to a `.env` file in this folder.")
        st.code("GEMINI_API_KEY=AIzaSy…", language="bash")
        st.stop()

    from ai_coach import generate_program, adapt_program, ask_coach

    ai_gen, ai_adapt, ai_chat = st.tabs(["🎯 Generate Program", "🔄 Adapt Program", "💬 Ask Coach"])

    # ── AI: Generate Program ──────────────────────────────────────────────────

    with ai_gen:
        st.markdown(
            f"**Profile:** {user['name']} · {user['goal']} · {user['fitness_level']} · "
            f"{user['age']} y · {user['weight']} kg"
        )
        c1, c2 = st.columns(2)
        weeks        = c1.slider("Duration (weeks)", 4, 16, 8)
        model_choice = c2.selectbox(
            "AI model",
            ["gemini-2.0-flash — fast & free ✅", "gemini-1.5-pro — best quality"],
        )
        model = "gemini-2.0-flash" if "flash" in model_choice else "gemini-1.5-pro"
        st.caption("💡 Gemini Flash is free-tier eligible — recommended for most users.")

        if st.button("🚀 Generate My AI Program", type="primary", use_container_width=True):
            with st.spinner("Analysing your profile and building a science-based program… (15–30 s)"):
                try:
                    prog = generate_program(user, weeks, model)
                    save_program(st.session_state.uid, prog.get("program_name", "AI Program"), prog, is_ai=True)
                    st.success(f"✅ **{prog.get('program_name')}** generated and saved to My Programs!")
                    st.caption(prog.get("description", ""))

                    if prog.get("tips"):
                        with st.expander("💡 Coach Tips"):
                            for tip in prog["tips"]:
                                st.write(f"• {tip}")

                    for week in prog.get("weeks", [])[:2]:
                        with st.expander(f"Week {week['week']}  — {week.get('notes', '')}"):
                            days = week.get("days", [])
                            if isinstance(days, str):
                                st.caption(days)
                            else:
                                for day in days:
                                    st.markdown(f"**{day.get('day', '')}** — {day.get('focus', '')}")
                                    for ex in day.get("exercises", []):
                                        if "duration_min" in ex:
                                            st.write(f"  • {ex['name']}: {ex.get('duration_min')} min ({ex.get('intensity', '')})")
                                        else:
                                            st.write(
                                                f"  • {ex['name']}: {ex.get('sets')} × {ex.get('reps')}"
                                                f"  — rest {ex.get('rest_sec', 60)} s"
                                            )
                except Exception as e:
                    st.error(f"Error generating program: {e}")

    # ── AI: Adapt Program ─────────────────────────────────────────────────────

    with ai_adapt:
        st.markdown("#### Adapt After Missed Sessions")
        st.caption("Tell the AI how many sessions you missed — it will intelligently reshuffle your upcoming week.")

        ai_programs = [p for p in get_programs(st.session_state.uid) if p["is_ai"]]

        if not ai_programs:
            st.info("Generate an AI program first, then come back here to adapt it.")
        else:
            prog_map  = {f"{p['name']}  ({p['created_at'][:10]})": p for p in ai_programs}
            sel_name  = st.selectbox("Program to adapt", list(prog_map.keys()))
            sel_prog  = prog_map[sel_name]

            c1, c2 = st.columns(2)
            missed = c1.number_input("Sessions missed", 1, 10, 1)
            reason = c2.text_input("Reason (optional)", placeholder="travel, injury, sick…")

            if st.button("🔄 Get Adaptation Plan", type="primary", use_container_width=True):
                with st.spinner("Analysing and adapting your program…"):
                    try:
                        adaptation = adapt_program(user, sel_prog, missed, reason)
                        st.success(adaptation.get("message", "Here is your adapted plan!"))

                        if adaptation.get("adjustments"):
                            st.subheader("Adjustments")
                            for adj in adaptation["adjustments"]:
                                with st.container(border=True):
                                    st.markdown(f"**{adj.get('day', '')}**: {adj.get('change', '')}")
                                    st.caption(adj.get("reason", ""))

                        if adaptation.get("modified_days"):
                            st.subheader("Modified Schedule")
                            for day in adaptation["modified_days"]:
                                with st.expander(f"{day.get('day', 'Day')} — {day.get('focus', '')}"):
                                    for ex in day.get("exercises", []):
                                        if "duration_min" in ex:
                                            st.write(f"• {ex['name']}: {ex.get('duration_min')} min")
                                        else:
                                            st.write(f"• {ex['name']}: {ex.get('sets')} × {ex.get('reps')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── AI: Chat ──────────────────────────────────────────────────────────────

    with ai_chat:
        st.markdown("#### 💬 Ask Your AI Coach")
        st.caption("Training, nutrition, recovery, form — ask anything. Keeps last 4 messages for context.")

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input("Ask your coach…"):
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        recent = get_sessions(st.session_state.uid, limit=5)
                        response = ask_coach(user, question, st.session_state.chat, recent)
                        st.markdown(response)
                        st.session_state.chat.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.chat:
            if st.button("🗑️ Clear chat history"):
                st.session_state.chat = []
                st.rerun()
