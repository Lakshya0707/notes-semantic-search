import os 

notes_folder = "notes_folder"
documents =[]
# content of the file will be saved here 

filenames = []
# file name will be saved here 

for filename in os.listdir(notes_folder):
    # sirf .txt wali files lo, baaki ignore karo
    if filename.endswith(".txt"):
        filepath = os.path.join(notes_folder, filename)
        # file ka pura path banaya, jaise "notes/note1.txt"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            # file ka pura content ek string mein padh liya
        

            documents.append(content)
            # content ko list mein daal diya
        
            filenames.append(filename)
            # filename ko bhi list mein daal diya

# # check karne ke liye print karo
# print(f"Total {len(documents)} notes loaded")
# print(filenames)

def chunk_text(text, chunk_size=50, overlap=10):
    # text ko words mein todo
    words = text.split()
    # ek list banayi har word alag-alag
    
    chunks = []
    # yahan saare chunks store honge
    
    start = 0
    # kahan se chunk shuru karna hai, uska pointer
    
    while start < len(words):
        end = start + chunk_size
        # chunk kahan khatam hoga (start se chunk_size words aage)
        
        chunk = " ".join(words[start:end])
        # start se end tak ke words ko wapas sentence banaya
        
        chunks.append(chunk)
        # chunk ko list mein daala
        
        start = end - overlap
        # agla chunk thoda overlap ke saath shuru hoga
        # taaki beech mein context na toote (jaise ek sentence do chunks mein na bat jaaye)
    
    return chunks

# ab har document (note) ko chunks mein todo
all_chunks = []
# yahan saare chunks (sab notes ke) ek saath store honge

chunk_sources = []
# har chunk kis file se aaya, uska naam yahan store hoga

for doc, fname in zip(documents, filenames):
    # documents aur filenames dono list ek saath loop mein chalayi (zip se)
    
    doc_chunks = chunk_text(doc)
    # is note ke chunks banaye
    
    for chunk in doc_chunks:
        all_chunks.append(chunk)
        # har chunk ko master list mein daala
        
        chunk_sources.append(fname)
        # us chunk ka source filename bhi save kiya

# print(f"Total {len(all_chunks)} chunks bane {len(documents)} notes se")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(all_chunks).tolist()
# ab documents ki jagah all_chunks ka embedding banaya - ye important change hai

# print(f"Total {len(embeddings)} embeddings bane")
# print(f"Har embedding ki length: {len(embeddings[0])}")

import chromadb
# ChromaDB import kiya - ye hamara vector database hai

client = chromadb.Client()
# ek in-memory ChromaDB client banaya
# (program band hote hi data gayab ho jayega - abhi practice ke liye theek hai)

collection = client.create_collection(name="my_notes")
# ek "collection" banaya - socho ye ek table/box hai jisme hum data rakhenge

ids = [str(i) for i in range(len(all_chunks))]
# har chunk ko ek unique ID di, jaise "0", "1", "2"...
# ChromaDB ko har entry ke liye ek ID chahiye hoti hai, string format mein

metadatas = [{"source": source} for source in chunk_sources]
# har chunk ke liye ek metadata dictionary banayi
# isme bas itna hai ki ye chunk kis file se aaya
# baad mein search result ke saath ye bhi dikha payenge "ye kis note se mila"

collection.add(
    ids=ids,                  # unique IDs
    documents=all_chunks,     # asli text (chunks)
    embeddings=embeddings,    # Step 4 mein bane vectors
    metadatas=metadatas       # extra info (source filename)
)
# sab kuch ek saath ChromaDB mein daal diya

# print(f"{len(all_chunks)} chunks ChromaDB mein store ho gaye")

def search_notes(query, n_results=3):
    # user ka sawaal (query) leta hai aur top matching chunks dhundta hai
    
    query_embedding = model.encode([query]).tolist()
    # query ko bhi usi model se vector mein convert kiya
    # [query] isliye list mein, kyuki encode() ek list expect karta hai
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
        # kitne top results chahiye (default 3)
    )
    
    return results
    # ChromaDB se mile results wapas bhej diye


def print_results(results):
    # results ko achhe se, readable format mein print karta hai
    
    documents = results["documents"][0]
    # matching chunks ka text
    
    metadatas = results["metadatas"][0]
    # un chunks ka source (filename)
    
    distances = results["distances"][0]
    # kitna "door/paas" hai query se (kam distance = zyada relevant)
    
    for i in range(len(documents)):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {metadatas[i]['source']}")
        print(f"Distance: {distances[i]:.4f}")
        print(f"Text: {documents[i]}")


print("\n=== Semantic Search ready hai! ===")
print("Query likho (exit likhne se band ho jayega)\n")

while True:
    # infinite loop - jab tak user "exit" na likhe
    
    query = input("Tumhara sawaal: ")
    # user se input lena
    
    if query.lower() == "exit":
        # agar user "exit" ya "Exit" ya "EXIT" kuch bhi likhe, chhota-bada sab chalega
        print("Bye bhai!")
        break
        # loop se bahar nikal jao, program band ho jayega
    
    if query.strip() == "":
        # agar user khali Enter daba de, kuch na likhe
        print("Kuch toh likho bhai!")
        continue
        # loop ke start pe wapas chala jao, neeche ka code skip karke
    
    results = search_notes(query)
    print_results(results)
    print("\n" + "-"*50 + "\n")
    # ek separator line dikhayi, taaki results alag-alag dikhein