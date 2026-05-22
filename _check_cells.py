import json
with open('G:/FAX/L2_IAD/S4/EXAMEN ML/projet_detection_intrusion.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

all_ok = True
for idx, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    status = ''
    if cell_type == 'code':
        src = ''.join(cell['source'])
        try:
            compile(src, f'cell_{idx}', 'exec')
            status = ' [OK]'
        except SyntaxError as e:
            status = f' [ERR ligne {e.lineno}]: {e.msg}'
            all_ok = False
            lines = src.split('\n')
            if e.lineno and e.lineno <= len(lines):
                status += f'\n   -> {repr(lines[e.lineno-1][:120])}'
    print(f'Cell {idx}: {cell_type}{status}')

print(f'\nTotal: {len(nb["cells"])} cells')
print(f'All OK: {all_ok}')
