import unicodedataplus as ud
from collections import Counter

def detect_script(text):
    script_counts = Counter()
    limit = 50

    for char in text[:limit]:
        if char.strip():
            try:
                script = ud.script(char)
                script_counts[script] += 1
            except Exception as e:
                print(f"Error getting script for character {char}: {e}")

    if not script_counts:
        return 'Unknown'

    most_common_script = script_counts.most_common(1)[0][0]
    return most_common_script
