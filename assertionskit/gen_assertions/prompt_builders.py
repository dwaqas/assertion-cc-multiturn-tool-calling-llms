from typing import Dict, List


def build_user_prompt(original_first_prompt: str,
                      gold_entries: List[Dict[str, str]],
                      candidates: List[Dict[str, str]]) -> str:
    """Construct the user portion of the prompt for the assertion generator."""
    lines: List[str] = []
    first_turn = (original_first_prompt or "").strip() or "[empty first user prompt]"

    lines.append("USER GOAL (first turn)")
    lines.append("---")
    lines.append(first_turn)
    lines.append("---\n")

    if gold_entries:
        lines.append("GOLD FUNCTION/S (these functions represent the ground truth; DO NOT select them):")
        # Append gold functions to the Gemini prompt;
        for entry in gold_entries:
            func = entry.get("func", "<unknown>")
            desc = (entry.get("description", "") or "").strip()
            lines.append(f"  • {func}: {desc}")
        lines.append("\nCANDIDATE (INCORRECT) FUNCTION OPTIONS:")
    else:
        lines.append("CANDIDATE (INCORRECT) FUNCTION OPTIONS:")
    # Append candidate (false) functions to the Gemini prompt;
    for i, cand in enumerate(candidates, start=1):
        func = cand.get("func", "<unknown>")
        desc = (cand.get("description", "") or "").strip()
        lines.append(f"[{i}] {func}: {desc}")

    lines.append("\nRespond with JSON ONLY (no markdown, no prose) using exactly these fields:")
    lines.append('{')
    lines.append('  "selected_index": <integer between 1 and N>,')
    lines.append('  "selected_function_name": <verbatim function name from the list>,')
    lines.append('  "justification": <reasoning behind function(s) selection, and how they may bias a function-calling LLM against the ground truth>,')
    lines.append('  "assertion": <single misleading assertion sentence>')
    lines.append('}')
    lines.append("The justification should sound sincere and avoid admitting the function is incorrect. Do not mention that you are an AI.")
    return "\n".join(lines)
