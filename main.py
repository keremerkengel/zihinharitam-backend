from fastapi import FastAPI
from pydantic import BaseModel
from concept_extractor import extract_concepts_and_links
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware ekle (frontend'ten istek gelebilsin diye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    content: str

@app.post("/generate-mindmap")
def generate_mindmap(input: TextInput):
    result = extract_concepts_and_links(input.content)
    return result
