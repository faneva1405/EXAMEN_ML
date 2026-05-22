#!/usr/bin/env python3
"""Script de correction de toutes les erreurs du projet."""
import json
import os

NOTEBOOK_PATH = r'G:\FAX\L2_IAD\S4\EXAMEN ML\projet_detection_intrusion.ipynb'

print("=" * 60)
print("  CORRECTION DU NOTEBOOK")
print("=" * 60)

with open(NOTEBOOK_PATH, encoding='utf-8') as f:
    nb = json.load(f)

fixes_count = 0

# Fix 1: Indentation bug - axes[i].set_xlabel outside for loop
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    for i, line in enumerate(cell['source']):
        # The line "axes[i].set_xlabel(...)" should be indented inside the for loop
        if line.startswith('axes[i].set_xlabel'):
            cell['source'][i] = '    ' + line
            fixes_count += 1
            print(f"  [FIX {fixes_count}] Cell {cell_idx}, line {i}: Added indentation to axes[i].set_xlabel")

# Fix 2: Chinese character in markdown cell
for cell_idx, cell in enumerate(nb['cells']):
    for i, line in enumerate(cell['source']):
        if '\u51b3\u7b56' in line:  # Chinese chars for "decision"
            cell['source'][i] = line.replace(
                "Ensemble de\u51b3\u7b56 arbres, robuste",
                "Ensemble d'arbres de d\u00e9cision, robuste"
            )
            fixes_count += 1
            print(f"  [FIX {fixes_count}] Cell {cell_idx}, line {i}: Replaced Chinese characters with French text")

# Save notebook
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n  Total notebook fixes: {fixes_count}")

# Verify notebook syntax after fixes
print("\n  Verifying syntax...")
all_ok = True
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        try:
            compile(src, f'cell_{idx}', 'exec')
        except SyntaxError as e:
            print(f"  [ERROR] Cell {idx}: {e.msg} at line {e.lineno}")
            all_ok = False

if all_ok:
    print("  All cells syntax OK!")
else:
    print("  SYNTAX ERRORS FOUND - check above")

# Ensure figures directory exists
os.makedirs(r'G:\FAX\L2_IAD\S4\EXAMEN ML\figures', exist_ok=True)
print("\n  figures/ directory ensured")

print("\n" + "=" * 60)
print("  DONE - All fixes applied")
print("=" * 60)
