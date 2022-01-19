import pandas as pd

def write_file(lines, fname):
    with open(fname, "w") as f:
        f.write("\n".join([l.replace("\n", " ") for l in lines]))
    print("wrote {}".format(fname))

aae_scores_file = "./prompts/aave_samples_scores.jsonl"
muse_scores_file = "./prompts/sae_samples_scores.jsonl"

aae_df = pd.read_json(aae_scores_file, lines=True)
muse_df = pd.read_json(muse_scores_file, lines=True)

aae_df = aae_df.rename(columns={'text':'aae_text', 'toxicity': 'aae_toxicity'})
muse_df = muse_df.rename(columns={'text':'muse_text', 'toxicity': 'muse_toxicity'})

cat_df = pd.concat([aae_df, muse_df], axis=1)

cat_df_toxic = cat_df[(cat_df['aae_toxicity'] >= 0.5) & (cat_df['muse_toxicity'] >= 0.5)]
cat_df_nontoxic = cat_df[(cat_df['aae_toxicity'] < 0.5) | (cat_df['muse_toxicity'] < 0.5)]

aae_toxic = "./realprompts/toxic_aae.txt"
aae_nontoxic = "./realprompts/nontoxic_aae.txt"
muse_toxic = "./realprompts/toxic_muse.txt"
muse_nontoxic = "./realprompts/nontoxic_muse.txt"

write_file(list(cat_df_toxic['aae_text']), aae_toxic)
write_file(list(cat_df_toxic['muse_text']), muse_toxic)
write_file(list(cat_df_nontoxic['aae_text']), aae_nontoxic)
write_file(list(cat_df_nontoxic['muse_text']), muse_nontoxic)
