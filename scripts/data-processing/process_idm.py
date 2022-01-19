import pandas as pd
import os

def write_file(lines, fname):
    with open(fname, "w") as f:
        f.write("\n".join([l.replace("\n", " ") for l in lines]))
    print("wrote {}".format(fname))

SOURCE_FILE = "./identity_mentions_scores.jsonl"
OUT_FILE = "./nontoxic_tweets.txt"

df = pd.read_json(SOURCE_FILE, lines=True)
toxic_df = df[df['toxicity'] >= 0.5].head(1000)
nontoxic_df = df[df['toxicity'] < 0.5].head(1000)
write_file(nontoxic_df['text'], OUT_FILE)
#toxic_chunks = [toxic_df[i:i + len(toxic_df)//4] for i in range(0,len(toxic_df),len(toxic_df)//4)]
#nontoxic_chunks = [nontoxic_df[i:i + len(nontoxic_df)//4] for i in range(0,len(nontoxic_df),len(nontoxic_df)//4)]

#for i, chunk in enumerate(toxic_chunks):
#    write_file(chunk['prompt.text'], os.path.join(folder, "toxic_prompts{}.txt".format(i + 1)))
#    write_file(chunk['continuation.text'], os.path.join(folder, "toxic_conts{}.txt".format(i + 1)))

#for i, chunk in enumerate(nontoxic_chunks):
#    write_file(chunk['prompt.text'], os.path.join(folder, "nontoxic_prompts{}.txt".format(i + 1)))
#    write_file(chunk['continuation.text'], os.path.join(folder, "nontoxic_conts{}.txt".format(i + 1)))

#write_file(toxic_df['prompt.text'],TOXIC_FILE)
#write_file(toxic_df['continuation.text'], TOXIC_CONTINUATIONS)
#write_file(nontoxic_df['prompt.text'],NONTOXIC_FILE)
#write_file(nontoxic_df['continuation.text'],NONTOXIC_CONTINUATIONS)
