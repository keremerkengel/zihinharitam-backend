from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

turkish_stopwords = set([
    "bir", "ve", "ile", "bu", "şu", "da", "de", "gibi", "olarak", "için", "ama",
    "fakat", "veya", "ya", "yada", "çünkü", "ki", "daha", "çok", "az", "en",
    "olan", "olanlar", "vardır", "vardir", "birçok", "birisi", "yani", "ise",
    "ancak", "dolayı", "dolayısıyla", "neden", "sonra", "önce", "şekilde", "etkisi"
])

def clean_keyword(kw):
    kw = kw.lower()
    kw = re.sub(r'[^\w\s]', '', kw)
    kw = kw.replace("â", "a").replace("î", "i").replace("û", "u")

    words = kw.split()
    words = [w for w in words if w not in turkish_stopwords and len(w) > 2]

    root_words = []
    for word in words:
        word = re.sub(r'(ların|lerin|dan|den|nin|nın|dir|dır|lik|lık|in|un|an|en)$', '', word)
        root_words.append(word)

    result = " ".join(root_words).strip()
    if len(result) < 3 or result.count(" ") > 2:
        return ""
    return result

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT(model=model)

def extract_concepts_and_links(text, top_n=8):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words=None,
        top_n=top_n
    )

    nodes = list({clean_keyword(kw[0]) for kw in keywords if clean_keyword(kw[0]) != ''})

    embeddings = model.encode(nodes)
    sim_matrix = cosine_similarity(embeddings)

    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            sim_score = sim_matrix[i][j]
            if sim_score > 0.6:
                edges.append({
                    "source": nodes[i],
                    "target": nodes[j],
                    "relation": "ilişkili (%.2f)" % sim_score
                })

    return {
        "nodes": nodes,
        "edges": edges
    }
