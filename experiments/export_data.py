import argparse 
import glob, os 
import json

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, help='where the experiment was run for the pkl files', default="/home/alex/drake-tamp/experiments/jobs/random-perception-large-05")
    parser.add_argument("--experiment_description", type=str, help='name for the experiment', default="rgbd3")
    parser.add_argument("--split", type=float, help='proportion of data used for training, rest for val', default=1)
    parser.add_argument("--export_dir", type=str, help='where to save the generated json', default="/home/alex/drake-tamp/learning/data/experiments/perception-large-05.json")
    return parser 

def get_pkl_files(experiment_dir: str) -> list:
    """returns a list of the pickle files from the experiment"""

    return glob.glob(experiment_dir + "/*/oracle/*/*_labels.pkl")
        

if __name__ == "__main__":
    args = make_argument_parser().parse_args()
    if args.split == -1:
        test_pkl_list = [os.path.join(os.getcwd(), pkl) for pkl in glob.glob(args.experiment_dir + "/test/oracle/*/*_labels.pkl")] 
        train_pkl_list = [os.path.join(os.getcwd(), pkl) for pkl in glob.glob(args.experiment_dir + "/train/oracle/*/*_labels.pkl")] 
        data = {
            "experiment_description":args.experiment_description, 
            "train":train_pkl_list,
            "validation":test_pkl_list
        }
    else:
        pkl_list = [os.path.join(os.getcwd(), pkl) for pkl in get_pkl_files(args.experiment_dir)] 
        train_end_index = int(args.split*len(pkl_list))
        data = {
            "experiment_description":args.experiment_description, 
            "train":pkl_list[:train_end_index],
            "validation":pkl_list[train_end_index:]
            }
    with open(args.export_dir, 'w+') as outfile:
        json.dump(data, outfile, indent=4)

