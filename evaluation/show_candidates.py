import json

with open("candidates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

candidates = data.get("candidates", data)

print("=" * 100)
print("TOP 10 CANDIDATES")
print("=" * 100)

for i, x in enumerate(candidates[:10], 1):
    print()
    print(f"#{i}")
    print(f"Product ID: {x.get('product_id')}")
    print(f"Title:      {x.get('title')}")
    print(f"Modality:   {x.get('modality')}")
    print(f"Text score: {x.get('raw_text_score')}")
    print(f"Image score:{x.get('raw_image_score')}")