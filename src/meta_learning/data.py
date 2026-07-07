import json
import random
import itertools

class Data():

    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path
        self.meta_tasks_frac = args.meta_tasks_frac
        self.meta_tasks_data_frac = args.meta_tasks_data_frac

    def prepare_data(self):
        if self.dataset_name == "geneval":
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f if line.strip()]
            tags = list(set([data_pt["tag"] for data_pt in data]))
        elif self.dataset_name == "T2I-CompBench":
            pass
        elif self.dataset_name == "Wise":
            pass
        num_tags = len(tags)
        # Sample tasks for meta-optimization
        meta_tasks = random.sample(tags, k=int(num_tags*self.meta_tasks_frac))
        # Get datapts corresponding to meta-tasks
        meta_data = list(filter(lambda x: x["tag"] in meta_tasks, data))
        meta_data_dict = {}
        for meta_datapt in meta_data:
            if meta_datapt['tag'] in meta_data_dict:
                meta_data_dict['tag'].append(meta_datapt)
            else:
                meta_data_dict['tag'] = [meta_datapt]
        self.meta_data_train = list(
            itertools.chain.from_iterable(
                [random.sample(task_meta_data, int(self.meta_tasks_data_frac*len(task_meta_data))) for _, task_meta_data in meta_data_dict]
                )
            )
        self.meta_data_test = [meta_datapt for meta_datapt in meta_data if meta_datapt not in self.meta_data_train]

    def get_data(self, mode="train"):
        if mode == "train":
            for train_pt in self.meta_data_train:
                yield train_pt
        else:
            for test_pt in self.meta_data_test:
                yield test_pt