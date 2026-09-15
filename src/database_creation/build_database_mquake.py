import argparse
import json
from data import get_dataset
from datasets import Dataset


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rollouts using LMLM agent with database lookups")
    parser.add_argument("--nb-examples", type=int, required=True, help="Number of examples to process, -1 for all")
    parser.add_argument("--using-triples", type=str, required=True, 
                        help="Use original triplets, e.g orig_triples_labeled, else it uses edited triplets, " \
                        "e.g new_triples_labeled  ", choices=["original", "new"])
    parser.add_argument("--database-save-dir", type=str, required=True, help="database save dir")
    args = parser.parse_args(argv)
    return args


def _get_triplets(dataset : Dataset, label : str):
    result = []
    for example in dataset:
        result.extend(example[label])
    return result

def main(args : argparse.Namespace):
    ds = get_dataset(name = "mquake-remastered", split = 'all')

    if args.using_triples == "original":
        label = "orig_triples_labeled"
    if args.using_triples == "new":
        label = "new_triples_labeled"
    
    triplets = _get_triplets(ds, label)

    database = {
        "entities" : [t[0] for t in triplets],
        "relationships" : [t[1] for t in triplets],
        "return_values" : [t[2] for t in triplets],
        "triplets" : triplets,
    }

    file_name = "mquake_remastered_cf6334_database.json"
    full_path = args.database_save_dir + "/" + file_name
    with open(full_path, "w") as f:
        json.dump(database, f, indent = 4)
    print("Saved mquake database to ", full_path)

if __name__ == '__main__':
    args = parse_args()
    main(args)