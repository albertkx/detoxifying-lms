import pandas as pd
import os
import sys

src_folder = sys.argv[1]
out_folder = sys.argv[2]

def write_file(lines, fname):
    with open(fname, "w") as f:
        f.write("\n".join([l.replace("\n", " ") for l in lines]))
    print("wrote {}".format(fname))

aae_df = pd.read_json(os.path.join(src_folder, "aave_samples_scores.jsonl"), lines=True)
aae_df = aae_df.rename(columns={'text': 'AAE_text', 'toxicity': 'AAE_toxicity'})
muse_df = pd.read_json(os.path.join(src_folder, "wae_samples_scores.jsonl"), lines=True)
muse_df = muse_df.rename(columns={'text': 'WAE_text', 'toxicity': 'WAE_toxicity'})
cat = pd.concat([aae_df, muse_df], axis=1)

nontoxic_df = cat[(cat['AAE_toxicity'] < 0.5) | (cat['WAE_toxicity'] < 0.5)]
toxic_df = cat[(cat['AAE_toxicity'] > 0.5) & (cat['WAE_toxicity'] > 0.5)]

# Write the full sentences
write_file(nontoxic_df["AAE_text"], os.path.join(out_folder, "nontoxic_aae.txt"))
write_file(nontoxic_df["WAE_text"], os.path.join(out_folder, "nontoxic_wae.txt"))
write_file(toxic_df["AAE_text"], os.path.join(out_folder, "toxic_aae.txt"))
write_file(toxic_df["WAE_text"], os.path.join(out_folder, "toxic_wae.txt"))
