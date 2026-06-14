import json
nb = json.load(open(r'C:\Users\grill\Documents\GitHub\cubemind\model\cubby\colab_v3_3_test.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if not src.strip():
        continue
    print(f"\n===== CELL {i} [{c['cell_type']}] =====")
    print(src)
