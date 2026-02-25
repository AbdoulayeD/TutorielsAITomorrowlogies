"""
AI Coach — powered by Anthropic Claude.
Premium features: program generation, session adaptation, coach chat.

Model choice: claude-haiku-4-5 (cheap, fast) vs claude-sonnet-4-6 (best quality).
Haiku is ~10x cheaper — recommended for most users.
"""
import json
from anthropic import Anthropic

client = Anthropic()

SYSTEM = (
    "You are FitAI, an expert fitness coach with deep knowledge in exercise science, "
    "strength & hypertrophy training, cardio, recovery, and evidence-based nutrition. "
    "You design programs based on current sports science (progressive overload, "
    "periodization, RIR/RPE, frequency-volume relationship). "
    "Be concise, practical, and motivating. Always return valid JSON when asked."
)


def generate_program(user: dict, weeks: int = 8, model: str = "claude-haiku-4-5-20251001") -> dict:
    """
    Generate a full N-week personalised training program.
    Returns a dict with program_name, description, weeks[], tips[].
    """
    prompt = f"""Create a {weeks}-week fitness program for:
- Age: {user['age']}, Weight: {user['weight']}kg, Height: {user['height']}cm
- Goal: {user['goal']}
- Fitness level: {user['fitness_level']}

Apply evidence-based principles: appropriate weekly volume (sets/muscle group),
progressive overload, exercise selection matching goal and level, deload week every 4th week.

Return ONLY valid JSON:
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
            {{"name": "Bench Press", "sets": 4, "reps": "6-8", "rest_sec": 120, "notes": "control descent"}},
            {{"name": "Running", "duration_min": 20, "intensity": "moderate", "type": "cardio"}}
          ]
        }},
        {{"day": "Tuesday", "focus": "Rest / Active Recovery", "exercises": []}}
      ]
    }}
  ],
  "tips": ["tip1", "tip2", "tip3"]
}}

Include all {weeks} weeks. Weeks 3+ may reference week 1 structure with progressive notes."""

    msg = client.messages.create(
        model=model,
        max_tokens=3500,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


def adapt_program(user: dict, program: dict, missed: int, reason: str = "", model: str = "claude-haiku-4-5-20251001") -> dict:
    """
    Suggest a smart 7-day adaptation after missed sessions.
    Redistributes volume, adjusts intensity, keeps the user on track.
    """
    content_preview = json.dumps(
        program.get("content", {}).get("weeks", [{}])[0], indent=2
    )[:900]

    prompt = f"""A user missed {missed} workout session(s).
Reason: {reason or "not specified"}
Profile: Goal = {user['goal']}, Level = {user['fitness_level']}

Current program (week 1 preview):
{content_preview}

Create a smart 7-day adaptation. Consider: redistribute volume, prioritize compound lifts,
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

    msg = client.messages.create(
        model=model,
        max_tokens=1400,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])


def ask_coach(user: dict, question: str, history: list = None, recent_sessions: list = None,
              model: str = "claude-haiku-4-5-20251001") -> str:
    """
    Conversational AI coach. Keeps last 4 messages for context.
    """
    profile = (
        f"User: {user['name']}, {user['age']}y, {user['weight']}kg | "
        f"Goal: {user['goal']} | Level: {user['fitness_level']}"
    )
    if recent_sessions:
        profile += f" | {len(recent_sessions)} recent sessions logged"

    messages = [
        {"role": "user", "content": f"My profile: {profile}"},
        {"role": "assistant", "content": "Got it! I have your profile. How can I help you today?"},
    ]

    # Append last 4 conversation turns for context
    for msg in (history or [])[-4:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    msg = client.messages.create(
        model=model,
        max_tokens=700,
        system=SYSTEM,
        messages=messages
    )
    return msg.content[0].text
