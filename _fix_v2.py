import json

with open('G:/FAX/L2_IAD/S4/EXAMEN ML/projet_detection_intrusion.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 5 (index 5) - find all lines with French apostrophes in single-quoted strings
for cell_idx in range(len(nb['cells'])):
    cell = nb['cells'][cell_idx]
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    modified = False
    for i, line in enumerate(src):
        # Find lines with patterns like '...d'...' that break Python
        # Check for French apostrophe inside single-quoted string
        stripped = line.strip()
        # Skip comments and empty lines
        if stripped.startswith('#') or not stripped:
            continue
        # Check if line has single-quoted string with French apostrophe
        # Look for pattern: '...d'...' 
        if "'d'" in line or "'D'" in line or "'l'" in line or "'L'" in line or "'s'" in line or "'S'" in line:
            # Check if this is a French contraction in a single-quoted Python string
            # Simple heuristic: if the char before 'd is a letter, it's French
            import re
            # Find patterns where we have a letter followed by ' followed by letters
            matches = list(re.finditer(r"(?<=[a-zA-Z])'[a-zA-Z]", line))
            if matches:
                # Check if we're inside a single-quoted Python string
                # Count single quotes before the match
                for m in matches:
                    pos = m.start()
                    before = line[:pos]
                    single_quotes = before.count("'")
                    double_quotes = before.count('"')
                    # If odd number of single quotes, we're inside a single-quoted string
                    if single_quotes % 2 == 1:
                        print(f"Cell {cell_idx} line {i}: {repr(line[:100])}")
                        # Fix: replace surrounding single quotes with double quotes for the string
                        # Strategy: find the outermost single-quoted string and change to double quotes
                        modified = True
    if modified:
        print(f"  -> Issues found in cell {cell_idx}")

# Now let's do a thorough scan and fix all French apostrophe issues
print("\n--- THOROUGH FIX ---")
for cell_idx in range(len(nb['cells'])):
    cell = nb['cells'][cell_idx]
    if cell['cell_type'] != 'code':
        continue
    src = cell['source']
    new_src = []
    changed = False
    for line in src:
        new_line = line
        # Pattern: set_title('...'...) or set_xlabel('...'...), etc.
        # where ... contains French apostrophe like d', l', s', n', etc.
        import re
        # Match: function_name('text with french'apostrophe', ...)
        # Replace with: function_name("text with french'apostrophe", ...)
        
        # Find all function calls with single-quoted strings that contain French apostrophes
        # Pattern: word( or .word( followed by '...text'...'
        func_match = re.finditer(r"(\w+)\s*\(\s*'([^']*)'", line)
        for m in func_match:
            func_name = m.group(1)
            inner_text = m.group(2)
            if "'" in inner_text:  # Has French apostrophe inside
                old = f"{func_name}('{inner_text}'"
                new = f'{func_name}("{inner_text}"'
                new_line = new_line.replace(old, new, 1)
                changed = True
                print(f"Cell {cell_idx}: Fixed {old[:60]} -> {new[:60]}")
        
        new_src.append(new_line)
    
    if changed:
        nb['cells'][cell_idx]['source'] = new_src

with open('G:/FAX/L2_IAD/S4/EXAMEN ML/projet_detection_intrusion.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nDone")
