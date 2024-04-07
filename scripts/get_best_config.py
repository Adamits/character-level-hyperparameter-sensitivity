import pandas as pd
import numpy as np

import os


NAME_MAP ={
    "albanian": "alb",
    "arabic": "ara",
    "bengali": "ben",
    "bulgarian": "bul",
    "catalan": "cat",
    "czech": "cze",
    "dutch": "dut",
    "french": "fre",
    "haida": "hai",
    "hungarian": "hun",
    "irish": "gle",
    "romanian": "rom",
}


def read_data(filepath: str) -> pd.DataFrame:
    """Reads the wandb runs into a DataFrame

    Args:
        filepath (str): Path to CSV file.

    Returns:
        pd.DataFrame.
    """
    data = pd.read_csv(filepath, sep="\t")
    print(f"Read {len(data)} runs")

    return data


def hparam_keys(data):
    smoothing_name = (
        "smoothing" if "smoothing" in data.columns else "label_smoothing"
    )
    return [
        "batch_size",
        "learning_rate",
        "beta1",
        "beta2",
        smoothing_name,
        "scheduler",
        "num_warmup_samples",
        "factor",
        "reduce_lr_patience",
        "min_lr",
        "encoder_layers",
        "hidden_size",
        "dropout",
        "embedding_size",
        "attention_heads",
        "decoder_layers",
    ]


def arch_hparam_keys():
    return [
        "encoder_layers",
        "hidden_size",
        "dropout",
        "embedding_size",
        "attention_heads",
        "decoder_layers",
    ]


def opt_hparam_keys():
    return [
        "batch_size",
        "learning_rate",
        "beta1",
        "beta2",
        "label_smoothing",
        "scheduler",
        "num_warmup_samples",
        "factor",
        "reduce_lr_patience",
        "min_lr",
    ]


def hparam_float_keys():
    return [
        "learning_rate",
        "beta1",
        "beta2",
        "label_smoothing",
        "factor",
        "min_lr",
        "dropout",
        "Acc",
    ]


def _modify_col_titles(titles):
    modified = []
    for t in titles:
        m = t.replace("_", " ")
        if m == "num warmup samples":
            m = "wrmp steps"
        if m == "learning rate":
            m = "LR"
        if m == "min lr":
            m = "min LR"
        if m == "label smoothing":
            m = "LS"
        if m == "encoder layers":
            m = "enc. layers"
        if m == "decoder layers":
            m = "dec. layers"
        if m == "hidden size":
            m = "hidden size"
        if m == "embedding size":
            m = "emb. size"
        if m == "reduce lr patience":
            m = "LR patience"
        if m == "attention heads":
            m = "attn heads"
        if m == "scheduler":
            m = "sched."

        modified.append(m)

    return modified


def _modify_val(val, k):
    if k == "learning_rate" or k == "min_lr":
        new_val = str(val)
        if val * 100 > 1:
            new_val = f"{round(val * 100, 2)}e-2"
        if val * 1000 > 1:
            new_val = f"{round(val * 1000, 2)}e-3"
        elif val * 10000 > 1:
            new_val = f"{round(val * 10000, 2)}e-4"
        elif val * 100000 > 1:
            new_val = f"{round(val * 100000, 2)}e-5"
        return new_val
    # elif k == "wrmp steps":
    #     return f"{round(val / 1000, 2)}k"

    if type(val) == np.float64:
        new_val = str(round(val, 2))
        if new_val == "0.0":
            new_val = str(round(val, 3))
        if new_val == "0.0":
            new_val = str(round(val, 4))
    else:
        new_val = str(val)
        if new_val == "reduceonplateau":
            new_val = "reduce"
        if new_val == "warmupinvsqrt":
            new_val = "wrmp."
        if new_val == "nan":
            new_val = "None"

    return new_val


def make_latex(df, acc_keys, caption, label, outpath):
    acc_str = "\\begin{table*}[t]"
    acc_str += "\n\t\\footnotesize"
    acc_str += "\n\t\\centering"
    acc_str += "\n\t\\begin{tabular}{ll|" + "r" * len(acc_keys) + "}"
    acc_str += "\n\t\\toprule"
    acc_str += (
        "\n\t"
        + "Task & Arch & "
        + " & ".join(_modify_col_titles(acc_keys))
        + "\\\\"
    )
    acc_str += "\n\t\\toprule"

    lang_idx = 0
    # TODO: FIrst task, then lang.
    for lang in sorted(set(df["Language"])):
        if lang_idx > 0:
            acc_str += "\n\t\\midrule"
        lang_idx += 1
        infl_arches = sorted(
            df.loc[(df["Task"] == "Infl.") & (df["Language"] == lang)]["Arch."]
        )
        if any(infl_arches):
            acc_str += (
                "\n\t\\multirow{"
                + str(len(infl_arches))
                + "}{*}{\\texttt{"
                + lang
                + "} Infl.} "
            )

        for arch in infl_arches:
            acc_str += "\n\t & " + arch
            match_df = df.loc[
                (df["Task"] == "Infl.")
                & (df["Arch."] == arch)
                & (df["Language"] == lang)
            ]

            for idx, row in match_df.iterrows():
                for k in acc_keys:
                    val = _modify_val(row[k], k)
                    acc_str += " & " + val

            acc_str += " \\\\"

    for lang in sorted(set(df["Language"])):
        g2p_arches = sorted(
            df.loc[(df["Task"] == "G2P") & (df["Language"] == lang)]["Arch."]
        )
        if any(g2p_arches):
            acc_str += "\n\t\\midrule"
            acc_str += (
                "\n\t\\multirow{"
                + str(len(g2p_arches))
                + "}{*}{\\texttt{"
                + lang
                + "} G2P} "
            )

        for arch in g2p_arches:
            acc_str += "\n\t & " + arch

            match_df = df.loc[
                (df["Task"] == "G2P")
                & (df["Arch."] == arch)
                & (df["Language"] == lang)
            ]
            if match_df.empty:
                acc_str += " & " * len(acc_keys)

            for idx, row in match_df.iterrows():
                for k in acc_keys:
                    val = _modify_val(row[k], k)
                    acc_str += " & " + val

            acc_str += " \\\\"

    acc_str += "\n\t\\bottomrule"
    acc_str += "\n\t\\end{tabular}"
    acc_str += "\n\t\\caption{" + caption + "}"
    acc_str += "\n\t\\label{" + label + "}"
    acc_str += "\n\\end{table*}"

    with open(outpath, "w") as out:
        print(acc_str, file=out)


def main(results_dir, output_filepath):
    all_bests = []
    for filename in os.listdir(results_dir):
        if "lstm" in filename:
            arch = "LSTM"
        elif "transformer" in filename:
            arch = "Trans."
        else:
            msg = "Cannot figure out architecture: neither `lstm` "
            msg += f"nor `transformer` is in the filename: {filename}"
            print(msg)
            continue
        # TODO: Update if we add tasks...
        if "g2p" in filename:
            task = "G2P"
        elif "inflection" in filename:
            task = "Infl."

        language = os.path.basename(filename).split("-")[0]
        language = NAME_MAP.get(language, language)
        print(language, task, arch)
        df = read_data(os.path.join(results_dir, filename))
        print(df.shape)
        if df.shape[0] < 2:
            continue
        df = df.sort_values("max_val_accuracy", ascending=False)
        best = df.iloc[0]
        hparams = hparam_keys(df)
        best = best[hparams + ["max_val_accuracy"]]
        best["Language"] = language
        best["Task"] = task
        best["Arch."] = arch
        best = best.to_frame().transpose().reset_index(drop=True)
        if "smoothing" in best.columns:
            best = best.rename({"smoothing": "label_smoothing"}, axis=1)
        # best = best.set_index(["Language", "Task", "Arch."])
        all_bests.append(best)

    # print([df.index for df in all_bests])
    all_df = pd.concat(all_bests).reset_index(drop=True)
    all_df.loc[
        all_df["scheduler"] != "reduceonplateau",
        ["factor", "min_lr", "reduce_lr_patience"],
    ] = 0
    all_df.loc[
        all_df["scheduler"] != "warmupinvsqrt", "num_warmup_samples"
    ] = 0
    all_df.loc[all_df["Arch."] == "LSTM", "attention_heads"] = 1
    all_df.to_csv(output_filepath)

    arch_keys = arch_hparam_keys()
    opt_keys = opt_hparam_keys()
    all_df = all_df.rename({"max_val_accuracy": "Acc"}, axis=1)
    all_df["Acc"] = all_df["Acc"] * 100
    # for hparam in hparams:
    #     print(hparam, all_df[hparam].dtype)
    #     all_df[hparam] = round(all_df[hparam], 2)
    caption = "Architectural hyperparameters for the best performing system in each task and language."
    label = "tab:best_arch_configs"
    # Changes warmup samples value to warmup steps.
    all_df.loc[:, "num_warmup_samples"] = (all_df["num_warmup_samples"] / all_df["batch_size"]).astype(int)
    make_latex(
        all_df,
        arch_keys,
        caption,
        label,
        output_filepath.replace(".csv", "_arch.tex"),
    )

    caption = "Optimization hyperparameters for the best performing system in each task and language."
    label = "tab:best_opt_configs"
    make_latex(
        all_df,
        opt_keys,
        caption,
        label,
        output_filepath.replace(".csv", "_opt.tex"),
    )


if __name__ == "__main__":
    main("results_oct", "tables/best_configs.csv")
