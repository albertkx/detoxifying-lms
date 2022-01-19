import sys
import numpy as np
import pandas as pd
import csv
import os
import click

@click.command()
@click.option("--path", type=click.Path(), help="path to kaggle input data")
@click.option("--ft-output", type=click.Path(), help="path to output ft data")
@click.option("--gedi-output", type=click.Path(), help="path to output gedi data")
@click.option("--pplm-output", type=click.Path(), help="path to output pplm data")
def main(path, pplm_output, gedi_output, ft_output):
    pd.set_option('mode.chained_assignment', None)
    input_df = pd.read_csv(path)
    print("Done reading")
    class_sample_df = input_df[["target", "comment_text"]]
    class_sample_df = class_sample_df[(class_sample_df.target >= 0.5) | (class_sample_df.target < 0.1)]
    class_sample_df["target"] = (class_sample_df["target"] >= 0.1).astype(int)
    class_sample_df["comment_text"] = class_sample_df["comment_text"].apply(lambda x: x.replace("\n", "").replace("\r", "").replace('\t', "")) 
    class_sample_df.to_csv(os.path.join(pplm_output, "train.tsv"), sep="\t", header=False, index=False)
    print("PPLM Data Done")
    class_sample_df_swapped = class_sample_df[["comment_text", "target"]]
    class_sample_df_swapped["target"] = class_sample_df_swapped["target"].apply(lambda x: 1 - x)
    gedi_train, gedi_valid = np.split(class_sample_df_swapped, [int(0.9*len(class_sample_df_swapped))])
    gedi_train.to_csv(os.path.join(gedi_output, "train.tsv"), sep="\t", header=False, index=False)
    gedi_valid.to_csv(os.path.join(gedi_output, "valid.tsv"), sep="\t", header=False, index=False)
    print("GeDi Data Done")
    finetuning_df = class_sample_df[class_sample_df.target == 0]
    finetuning_df = finetuning_df[["comment_text"]]
    ft_train, ft_valid = np.split(finetuning_df, [int(0.9*len(finetuning_df))])
    ft_train.to_csv(os.path.join(ft_output, "train.tsv"), sep="\t", header=False, index=False)
    ft_valid.to_csv(os.path.join(ft_output, "valid.tsv"), sep="\t", header=False, index=False)
    print("FT Data Done")

if __name__ == "__main__":
    main()
