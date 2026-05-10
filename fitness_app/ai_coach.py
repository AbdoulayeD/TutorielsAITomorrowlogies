"""
AI Coach — powered by Google Gemini Flash.
Premium features: program generation, session adaptation, coach chat.
"""
import json
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM = (
    "You are FitAI, an expert fitness coach with deep knowledge in exercise science, "
    "strength & hypertrophy training, cardio, recovery, and evidence-based nutrition. "
    "You design programs based on current sports science (progressive overload, "
    "periodization, RIR/RPE, frequency-volume relationship). "
    "Be concise, practical, and motivating. Always return valid JSON when asked."
)


def _chat(prompt: str, model: str = "gemini-2.0-flash", json_mode: bool = False) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        max_output_tokens=8192,
        **({"response_mime_type": "application/json"} if json_mode else {}),
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def generate_program(user: dict, weeks: int = 8, model: str = "gemini-2.0-flash") -> dict:
    prompt = f"""Create a {weeks}-week fitness program for:
- Age: {user['age']}, Weight: {user['weight']}kg, Height: {user['height']}cm
- Goal: {user['goal']}
- Fitness level: {user['fitness_level']}

Apply evidence-based principles: appropriate weekly volume (sets/muscle group),
progressive overload, exercise selection matching goal and level, deload week every 4th week.

Return ONLY valid JSON with this structure:
{{
  "program_name": "...",
  "description": "2-3 sentences",
  "weeks": [
    {{
      "week": 1,
      "notes": "focus / intensity note",
      "days": [
        {{
          "day": "Monday",
          "focus": "e.g. Push / Upper / Full Body",
          "exercises": [
            {{"name": "Bench Press", "sets": 4, "reps": "6-8", "rest_sec": 120}},
            {{"name": "Running", "duration_min": 20, "intensity": "moderate", "type": "cardio"}}
          ]
        }},
        {{"day": "Tuesday", "focus": "Rest / Active Recovery", "exercises": []}}
      ]
    }}
  ],
  "tips": ["tip1", "tip2", "tip3"]
}}

Include all {weeks} weeks. Limit each day to 6 exercises max to keep the response concise."""

    return json.loads(_chat(prompt, model, json_mode=True))


def adapt_program(user: dict, program: dict, missed: int, reason: str = "", model: str = "gemini-2.0-flash") -> dict:
    content_preview = json.dumps(
        program.get("content", {}).get("weeks", [{}])[0], indent=2
    )[:900]

    prompt = f"""A user missed {missed} workout session(s).
Reason: {reason or "not specified"}
Profile: Goal = {user['goal']}, Level = {user['fitness_level']}

Current program (week 1 preview):
{content_preview}

Create a smart 7-day adaptation. Redistribute volume, prioritize compound lifts,
adjust intensity if fatigued, keep progressive overload on track.

Return ONLY valid JSON:
{{
  "message": "brief encouraging message (1-2 sentences)",
  "adjustments": [
    {{"day": "Monday", "change": "description", "reason": "why"}}
  ],
  "modified_days": [
    {{
      "day": "Day name",
      "focus": "...",
      "exercises": [
        {{"name": "...", "sets": 3, "reps": "8-10", "rest_sec": 90}}
      ]
    }}
  ]
}}"""

    return json.loads(_chat(prompt, model, json_mode=True))


def ask_coach(user: dict, question: str, history: list = None, recent_sessions: list = None,
              model: str = "gemini-2.0-flash") -> str:
    profile = (
        f"User: {user['name']}, {user['age']}y, {user['weight']}kg | "
        f"Goal: {user['goal']} | Level: {user['fitness_level']}"
    )
    if recent_sessions:
        profile += f" | {len(recent_sessions)} recent sessions logged"

    history_text = ""
    for msg in (history or [])[-4:]:
        if msg.get("role") in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "Coach"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Profile: {profile}

{history_text}User: {question}"""

    return _chat(prompt, model)
