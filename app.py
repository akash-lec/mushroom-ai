import os
import json
import re

import gradio as gr
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Add it in Render Environment Variables.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
You are a mushroom expert chatbot.

Your role:
- Answer questions about mushrooms and fungi.
- Help with mushroom identification from images.
- Keep answers short, natural, and factual.
- Clearly mention uncertainty when identification is not reliable.

Safety:
- Never claim that a wild mushroom is safe to eat based only on an image.
- Toxic mushrooms may be discussed factually, including general effects and historical use.
- Do not give step-by-step preparation, detoxification, or consumption instructions.
- If a toxic mushroom has historically been consumed after special processing,
  you may mention that only in general terms without explaining the method.
- Make clear that misidentification or incorrect handling can be dangerous.
- Avoid inventing medical, chemical, or species information.

Topic control:
- Stay focused on mushrooms and fungi.
- If the user asks about an unrelated topic, politely redirect the conversation back to mushrooms.
"""


def calibrate_confidence(result):
    visible_count = len(result.get("visible", []))
    confidence = result.get("confidence", 0.0)

    if visible_count == 0:
        confidence = min(confidence, 0.30)
    elif visible_count == 1:
        confidence = min(confidence, 0.50)
    elif visible_count == 2:
        confidence = min(confidence, 0.70)

    result["confidence"] = confidence
    return result


def analyze_mushroom(image):
    prompt = """
Carefully analyze this mushroom image.

First inspect the visible cap, hymenium, stipe, color,
surface texture, and other useful visual features.

Then make the most likely identification.

Return ONLY valid JSON:

{
  "common_name": "string",
  "genus": "string",
  "confidence": 0.0,
  "visible": [],
  "color": "string",
  "edible": false
}

Rules:
- confidence must be between 0 and 1
- visible may only contain "cap", "hymenium", and "stipe"
- include only parts clearly visible in the image
- do not assume hidden features
- distinguish look-alike mushrooms carefully
- lower confidence when important features are missing
- use "Unknown" when the visual evidence is too weak
- color should describe the main visible color
- edible must be true or false based on general knowledge of the predicted species
- never use the edible field as advice that the photographed mushroom is safe to eat
- return JSON only
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
        ),
    )

    text = response.text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group())
            allowed_parts = {"cap", "hymenium", "stipe"}
            result["visible"] = [
                part for part in result.get("visible", []) if part in allowed_parts
            ]

            if not isinstance(result.get("confidence"), (int, float)):
                result["confidence"] = 0.0
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))

            if not isinstance(result.get("edible"), bool):
                result["edible"] = False

            result["common_name"] = result.get("common_name", "Unknown")
            result["genus"] = result.get("genus", "Unknown")
            result["color"] = result.get("color", "Unknown")

            return calibrate_confidence(result)
        except json.JSONDecodeError:
            pass

    return {
        "common_name": "Unknown",
        "genus": "Unknown",
        "confidence": 0.0,
