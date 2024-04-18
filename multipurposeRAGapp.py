from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.neighbors import NearestNeighbors
import json
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
import os
from pypdf import PdfReader
import gradio as gr

claude_api_key = None # ask isaac

def encode_docs(uploaded_files:list, encoding_folder:str, mode, progress=gr.Progress()):
    encoding_folder = os.getcwd() + f"/{encoding_folder}"
    print(encoding_folder)
    # creating folder and children
    Path(encoding_folder).mkdir(parents=True, exist_ok=True) # create dir if not exist
    (Path(encoding_folder) / "txts").mkdir(parents=True, exist_ok=True) # create txt folder
    (Path(encoding_folder) / "chunks").mkdir(parents=True, exist_ok=True) # create chunks folder

    # gathering all files specified in app
    uploaded_files = [] if uploaded_files is None else uploaded_files

    # coverting pdf -> txt, copying all files to new encoding folder
    for file in progress.tqdm(uploaded_files, desc="Loading Files"):
        file_name = file.split('/')[-1]
        file_stem = file_name.split('.')[-1]
        if file_stem == "pdf":
            reader = PdfReader(file)
            number_of_pages = len(reader.pages)
            text = ""
            for page_num in range(number_of_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
        if file_stem == "txt":
            text = open(file).read()
        file_name = file_name.split(".")[0] + ".txt"
        f = open(encoding_folder + f"/txts/{file_name}", "a")
        f.write(text)

    # load txt files into langchain
    progress(0, desc="Loading Langchain Text Splitter")
    loader = DirectoryLoader(Path(encoding_folder)/"txts", glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()


    # splitting documents into manageable-sized chunks (thanks langchain!)
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 100)
    document_chunks = text_splitter.split_documents(documents)

    document_chunks=[f"Context: {chunk.page_content} Source: {chunk.metadata['source']}" for chunk in progress.tqdm(document_chunks, desc="Splitting Documents")]

    # create embeddings for txt chunks
    progress(0, desc="Loading HuggingFace Encoder (may take several minutes)")
    embeddings = HuggingFaceEmbeddings() # load huggingface text embeddings (transform documents to numbers for later comparison)

    progress(0, desc="creating pandas dataframe")
    df = pd.DataFrame(document_chunks, columns =['text'])
    index_embeddings = []

    progress((0, len(df)), desc="generating embeddings")
    for i, (index, doc) in enumerate(df.iterrows()):
        finished = 0
        embedding = embeddings.embed_query(doc["text"])
        if embedding is not None:
            doc_id=f"{index}.txt"
            embedding_dict = {
                    "id": doc_id,
                    "embedding": [str(value) for value in embedding],
            }
            index_embeddings.append(json.dumps(embedding_dict) + "\n")
            doc_id = f"{index}.txt"
            with open(f"{encoding_folder}/chunks/{doc_id}", "w") as document:
                document.write(doc['text'])
        with open(Path(encoding_folder)/"embeddings.json", "w") as f:
            f.writelines(index_embeddings)
        finished += 1
        progress((i, len(df)), desc="generating embeddings")
    return True, embeddings

def load_encodings(encoding_folder:str, mode,  progress=gr.Progress()):
    encoding_folder = os.getcwd() + f"/{encoding_folder}"
    progress((0, 1), desc="reading encodings")
    embeddings_json = encoding_folder+"/embeddings.json" # getting embedded data from previous steps
    file = open(embeddings_json)
    line = file.readline()
    full_array = []
    while line: # loading embeddings into memory -> numpy array
        embed = json.loads(line)['embedding']
        full_array.append(embed)
        line = file.readline()
    embeddings_array = np.array(full_array, dtype=np.float32)

    # creating k nearest neighbors object
    n_neighbors = 8
    progress((0, 1), desc="clustering data")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(embeddings_array)

    tokenizer, model = None, None
    if mode:
        progress((0, 2), desc="loading text models")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large") # loading tokenizer for flan
        progress((1, 2), desc="loading text models")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large") # loading flan text generation model
    else:
        model = Anthropic(api_key=claude_api_key)

    return True, nbrs, tokenizer, model

def generate_response(question:str, encoding_folder, embeddings, nbrs, tokenizer, model, mode, progress=gr.Progress()):
    encoding_folder = os.getcwd() + f"/{encoding_folder}"
    if embeddings is None:
        progress((0,1), desc="loading embeddings")
        embeddings = HuggingFaceEmbeddings()
    progress((0,1), desc="embedding question")
    embedding = embeddings.embed_query(question) # embed question to latent space using huggingface embeddings
    progress((0,1), desc="getting relevant documents")
    distances, indices = nbrs.kneighbors([embedding]) # get 8 most similar

    file = open(encoding_folder+f"/chunks/{indices[0][0]}.txt", 'r')
    context = file.read()
    # for i in range(min([amount_of_context, len(indices[0]), n_neighbors] )): # iterate over each found document
    #     file = open(warframe_text_chunks_folder / f"{indices[0][i]}.txt", 'r')
    #     context += file.read() # open its text

    prompt=f"""
    Follow exactly these 3 steps:
    1. Read the context below and aggregrate this data
    2. Answer the question using only this context
    3. Show the source for your answers
    If you don't have any context and are unsure of the answer, reply that you don't know about this topic.
    Context : {context}
    User Question: {question}
    """
    progress((0,1), desc="prompting model")

    if mode:
        model_input = tokenizer(prompt, return_tensors="pt").input_ids # tokenizing prompt for model
        model_output = model.generate(model_input, min_length=100, max_length=2000) # generating response
        text_output = tokenizer.decode(model_output[0]) # decoding response
    else:

        message = model.messages.create(
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        # model="claude-3-opus-20240229",
        # model = "claude-3-sonnet-20240229"
        model = "claude-3-haiku-20240307",
        )
        return message.content[0].text
    return text_output

if __name__ == "__main__":
    with gr.Blocks() as demo:
        # "global" variables
        embeddings = gr.State(None)
        nbrs = gr.State(None)
        tokenizer = gr.State(None)
        model = gr.State(None)
        with gr.Column() as outer_col:
            with gr.Accordion(label="Instructions", open=False) as instructions:
                instructions = open(os.getcwd()+"/instructions.md").read()
                gr.Markdown(value=instructions)
            with gr.Row() as row:
                with gr.Column() as col1:
                    mode = gr.Checkbox(False, label=" run model locally (Google Flan) \n (if not checked, uses Anthropic Claude API)")
                    files = gr.Files(label="pdfs or txt files", file_types=[".pdf", ".txt"])
                    encodings_folder = gr.Textbox(label="Encodings Name", value="my_awesome_encodings")
                    encode_button = gr.Button("Create Document Encodings")
                    encode_check = gr.Checkbox(value=False, interactive=False, label="")
                    load_button = gr.Button("Load Document Encodings", scale=8)
                    load_check = gr.Checkbox(value=False, interactive=False, label="", scale=1, min_width=0)
                    encode_button.click(fn=encode_docs, inputs=[files, encodings_folder], outputs=[encode_check, embeddings])
                    load_button.click(fn=load_encodings, inputs=[encodings_folder, mode], outputs=[load_check, nbrs, tokenizer, model])
                with gr.Column() as col2:
                    user_input = gr.Textbox(label="Question")
                    generate_button = gr.Button("Generate Response")
                    textbox = gr.Text(value="...", label="Response")
                    generate_button.click(fn=generate_response, inputs=[user_input, encodings_folder, embeddings, nbrs, tokenizer, model, mode], outputs=[textbox])
    demo.launch()#share=True)
