from typing import Dict, List

def build_user_prompt(original_first_prompt: str,
                      candidates: List[Dict[str, str]]) -> str:
    lines = []
    lines.append("USER FIRST PROMPT:\n---")
    lines.append(original_first_prompt.strip())
    lines.append("---\n")
    lines.append("CANDIDATE WRONG FUNCTIONS (choose EXACTLY ONE; scores not shown).")
    for i, c in enumerate(candidates, start=1):
        lines.append(f"[{i}] {c.get('func','<unknown>')}  —  {c.get('description','').strip()}")
    # Tight JSON contract (no extra keys; no markdown);
    lines.append("\nReturn ONLY JSON with EXACT fields and types:")
    lines.append('{')
    lines.append('  "selected_index": <integer: 1..N>,')
    lines.append('  "selected_function_name": <string>,')
    lines.append('  "justification": <string>,')
    lines.append('  "assertion": <string>')
    lines.append('}')
    lines.append("Do not include extra keys; do not wrap in markdown.")
    return "\n".join(lines)
