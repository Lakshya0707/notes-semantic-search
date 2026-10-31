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

print(f"Total {len(embeddings)} embeddings bane")
print(f"Har embedding ki length: {len(embeddings[0])}")