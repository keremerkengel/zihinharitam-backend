from fastapi import FastAPI
from pydantic import BaseModel
from concept_extractor import extract_concepts_and_links

app = FastAPI()

class TextInput(BaseModel):
    content: str

@app.post("/generate-mindmap")
def generate_mindmap(input: TextInput):
    result = extract_concepts_and_links(input.content)
    return result
