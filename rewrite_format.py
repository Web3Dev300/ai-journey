import re

with open('04_data_science/DATA_SCIENCE_ANSWERS.md', 'r') as f:
    content = f.read()

pattern = re.compile(r'^(\d+)\.\s+\*\*(.*?)\*\*\s*(.*)$', re.MULTILINE)

def replacement(match):
    step_num = match.group(1)
    # Remove any trailing colons in the title
    title = match.group(2).strip().rstrip(':')
    text = match.group(3).strip()
    # If text is empty (like when the text is on the next lines), just return the title
    if text:
        return f"**Step {step_num}: {title}**\n{text}\n"
    else:
        return f"**Step {step_num}: {title}**\n"

new_content = pattern.sub(replacement, content)

with open('04_data_science/DATA_SCIENCE_ANSWERS.md', 'w') as f:
    f.write(new_content)
