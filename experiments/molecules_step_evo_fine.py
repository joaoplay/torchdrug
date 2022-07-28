import os
from collections import OrderedDict

import hydra
import wandb
from omegaconf import DictConfig
from torch import optim

from torchdrug import datasets, models, tasks
from torchdrug.core import Engine

USE_CUDA = int(os.getenv("USE_CUDA", 0))
WANDB_PATH = '/data' if USE_CUDA else '.'

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


@hydra.main(config_path="configs", config_name="default_config")
def run(cfg: DictConfig):

    dynamic_tasks = None
    if cfg.use_dynamic_task:
        print("Using dynamic task")
        dynamic_tasks = OrderedDict({
            300: ['qed'],
            600: ['plogp'],
            10000000000000: ['qed', 'plogp']
        })
        run_name = f'{cfg.run_name}-{list(dynamic_tasks.items())}'
    else:
        run_name = f'{cfg.run_name}-{cfg.task}'

    dynamic_size = None
    if cfg.use_dynamic_size:
        dynamic_size = OrderedDict({
            100: 10,
            200: 20,
            300: 30,
            10000000000000: 38
        })

    print(f"Run name: {run_name}")
    wandb.init(project="molecule-generation", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"),
               dir=WANDB_PATH, name=run_name)
    dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True, node_feature="symbol")
    model = models.RGCN(input_dim=dataset.node_feature_dim, num_relation=dataset.num_bond_type,
                        hidden_dims=[256, 256, 256, 256], batch_norm=False)
    task = tasks.GCPNGeneration(model, [6, 7, 8, 9, 15, 16, 17, 35, 53], max_edge_unroll=12, max_node=38,
                                task=cfg.task, criterion='ppo', reward_temperature=1, agent_update_interval=3, gamma=0.9,
                                dynamic_task=dynamic_tasks, dynamic_size=dynamic_size)
    optimizer = optim.Adam(task.parameters(), lr=1e-5)
    solver = Engine(task, dataset, None, None, optimizer, gpus=cfg.gpus, logger="wandb", batch_size=128, log_interval=1)

    solver.load(BASE_PATH + '/model.pth')

    solver.train(num_epoch=1)

    # results = task.generate(num_sample=32, max_resample=5)
    # results.visualize(num_row=4, num_col=None, save_file=None, titles=None)


if __name__ == '__main__':
    run()


