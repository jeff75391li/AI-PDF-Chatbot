import fitz
import re
from openai import OpenAI
from scipy import spatial

searcher = None


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text.strip()


def pdf_to_text(file):
    doc = fitz.open(stream=file.getvalue(), filetype="pdf")
    total_pages = doc.page_count

    # Each element represents one page of text
    texts = []
    for i in range(0, total_pages):
        text = doc.load_page(i).get_text()
        text = preprocess(text)
        texts.append(text)
    doc.close()
    return texts


def text_to_chunks(texts, word_length=200):
    words_in_text = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(words_in_text):
        for i in range(0, len(words), word_length):
            chunk = words[i: i + word_length]
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page Number {idx + 1}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class EmbeddingSearcher:
    def __init__(self, file_name, api_key):
        self.client = OpenAI(api_key=api_key)
        self.data = None
        self.embeddings = None
        self.file_name = file_name

    def load(self, data, batch_size=1000):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch_size)

    def find_top_k_info_by_relatedness(
            self,
            query,
            relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
            top_k=10):
        query_embedding_response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query)
        query_embedding = query_embedding_response.data[0].embedding
        info_and_relatedness = [
            (self.data[idx], relatedness_fn(query_embedding, emb))
            for idx, emb in enumerate(self.embeddings)
        ]
        info_and_relatedness.sort(key=lambda x: x[1], reverse=True)
        texts, _ = zip(*info_and_relatedness)
        top_k = min(top_k, len(self.data))
        return texts[:top_k]

    def get_text_embedding(self, texts, batch_size=1000):
        embeddings = []
        for batch_start in range(0, len(texts), batch_size):
            batch_end = batch_start + batch_size
            batch = texts[batch_start:batch_end]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch)
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings


def init_searcher(file_name, file, api_key):
    global searcher
    # We load the data from the input file when:
    # 1. the searcher has not yet initialized
    # 2. the input file is different with the one being loaded
    if searcher is None or searcher.file_name != file_name:
        searcher = EmbeddingSearcher(file_name, api_key)
        texts = pdf_to_text(file)
        chunks = text_to_chunks(texts)
        searcher.load(chunks)


def generate_prompt(question):
    top_k_texts = searcher.find_top_k_info_by_relatedness(question)
    prompt = "Search results:\n\n"
    for chunk in top_k_texts:
        prompt += chunk + '\n\n'

    prompt += (
        "Instructions: Compose a comprehensive reply to the query using the search results given. "
        "Cite each reference using [Page Number] notation (every result has this number at the beginning). "
        "Citation is done at the end of each sentence. The format of citation is '[number]' (for example, [25]). "
        "If the search results mention multiple subjects with the same name, "
        "create separate answers for each. Only include information found in the results and "
        "do not add any additional information. Make sure the answer is correct and do not output false content. "
        "If the text does not relate to the query, simply state 'I could not find an answer'. Ignore outlier "
        "search results which has nothing to do with the question. Only answer what is asked. The "
        "answer should be short and concise. Answer step by step.\n\n"
    )
    prompt += f"Query: {question}\nAnswer:"
    return prompt


def answer_question(question, api_key):
    prompt = generate_prompt(question)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    message = response.choices[0].message.content
    return message


def run_query(file_name, file, question, api_key):
    init_searcher(file_name, file, api_key)
    return answer_question(question, api_key)
