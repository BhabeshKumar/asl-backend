from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from symspellpy import SymSpell
from gramformer import Gramformer
import time
import os
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with exact domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class InferenceRequest(BaseModel):
    keypoints: list

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = MLPClassifier(input_dim=126, num_classes=26)
model.load_state_dict(torch.load("sign_model.pt", map_location=torch.device("cpu")))
model.eval()
label_map = {i: chr(65 + i) for i in range(26)}

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = "symspell/frequency_dictionary_en_82_765.txt"
if not sym_spell.load_dictionary(dict_path, term_index=0, count_index=1):
    raise FileNotFoundError("SymSpell dictionary not found.")

gf = Gramformer(models=1)

buffer = ""
cleaned = ""
prediction_buffer = []
buffer_size = 5
last_added_letter = ""
last_action_time = time.time()
space_interval = 2.5

def merge_single_letters(words):
    merged = []
    buffer = ""
    for word in words:
        if len(word) == 1 and word.isalpha():
            buffer += word
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(word)
    if buffer:
        merged.append(buffer)
    return merged

def smart_correct(sentence):
    s = sentence.strip()
    if not s or len(s) < 3:
        return sentence
    try:
        if all(c.isupper() for c in s) and " " not in s and len(s) > 3:
            s = " ".join(list(s))

        raw_words = s.split()
        words = merge_single_letters(raw_words)

        joined_input = " ".join(words)
        suggestions = sym_spell.lookup_compound(joined_input.lower(), max_edit_distance=2)
        result = suggestions[0].term if suggestions else joined_input

        corrected = gf.correct(result.lower(), max_candidates=1)
        final = next(iter(corrected)) if corrected else result

        if "uncategorized" in final.lower() or "permalink" in final.lower():
            return result.capitalize()

        return final[0].upper() + final[1:] if final else result
    except:
        return sentence

@app.post("/predict")
def predict(req: InferenceRequest):
    global buffer, cleaned, prediction_buffer, last_added_letter, last_action_time

    if len(req.keypoints) != 126:
        raise HTTPException(status_code=400, detail="Expected 126 keypoints")

    input_tensor = torch.tensor([req.keypoints], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    if confidence.item() >= 0.9:
        letter = label_map[pred_class.item()]
        prediction_buffer.append(letter)

        if len(prediction_buffer) > buffer_size:
            prediction_buffer.pop(0)

        if (len(prediction_buffer) == buffer_size and
            prediction_buffer.count(letter) == buffer_size and
            letter != last_added_letter):
            buffer += letter
            last_added_letter = letter
            last_action_time = time.time()
            prediction_buffer.clear()

    # insert space and trigger correction if idle
    if time.time() - last_action_time > space_interval and not buffer.endswith(" "):
        buffer += " "
        cleaned = smart_correct(buffer)
        last_action_time = time.time()
        last_added_letter = ""

    return {"status": "ok"}

@app.get("/buffer")
def get_buffer():
    return {"raw": buffer, "cleaned": cleaned}

@app.post("/reset")
def reset():
    global buffer, cleaned, prediction_buffer, last_added_letter
    buffer = ""
    cleaned = ""
    prediction_buffer.clear()
    last_added_letter = ""
    return {"status": "reset"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
