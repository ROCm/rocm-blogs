import random
import json

for _ in range(128):
    prompt = f"What is {random.randint(0, 1024)} times {random.randint(0, 1024)}?"
    print(json.dumps({"prompt": prompt}))

