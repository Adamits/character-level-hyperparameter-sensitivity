# TODO: read pred and gold files, compute acuracy, and print it.

def read(fname):
    D = []
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            fields = line.split("\t")
            # Predictions
            if len(fields) == 1:
                D.append(fields[0])
            # G2P
            elif len(fields) == 2:
                D.append(fields + [""])
            # Golds
            elif len(fields) == 3:
                D.append(fields)
            else:
                msg = f"Unexpected number of fields ({len(fields)}) in the file {fname}"
                raise Exception(msg)
    return D


def eval_form(gold, predicted, ignore=set()):
    """compute average accuracy and edit distance for task 1 """
    if not len(gold) == len(predicted):
        msg = f"Unequal number of golds ({len(gold)}) and predictions ({predicted})."
        raise Exception(msg)
    
    correct, total = 0., 0.
    for (lemma, word, tag), pred in zip(gold, predicted):
        if (lemma, tag) in ignore:
            continue
        if word == pred:
            correct += 1
        total += 1
    return round(correct/total*100, 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CoNLL-SIGMORPHON 2017 Shared Task Evaluation')
    parser.add_argument("--gold", help="Gold standard (uncovered)", required=True, type=str)
    parser.add_argument("--predicted", help="Model output", required=True, type=str)
    args = parser.parse_args()    

    D_gold = read(args.gold)
    D_predicted = read(args.predicted)
    print(eval_form(D_gold, D_predicted))