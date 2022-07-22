import os

import wandb
from torch import optim

from torchdrug import datasets, models, tasks
from torchdrug.core import Engine

USE_CUDA = int(os.getenv("USE_CUDA", 0))

os.environ["WANDB_API_KEY"] = '237099249b3c0e91437061c393ab089d03339bc3'

if __name__ == '__main__':
    WANDB_PATH = '/data' if USE_CUDA else '.'

    run = wandb.init(project="molecule-generation", entity="jbsimoes", mode=os.getenv("WANDB_UPLOAD_MODE", "online"), dir=WANDB_PATH)

    dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True, node_feature="symbol")
    model = models.RGCN(input_dim=dataset.node_feature_dim, num_relation=dataset.num_bond_type, hidden_dims=[256, 256, 256, 256], batch_norm=False)
    task = tasks.GCPNGeneration(model, [6, 7, 8, 9, 15, 16, 17, 35, 53], max_edge_unroll=12, max_node=38, task=["plogp", "qed"],
                                criterion="ppo", reward_temperature=1, agent_update_interval=3, gamma=0.9)
    optimizer = optim.Adam(task.parameters(), lr=1e-5)
    solver = Engine(task, dataset, None, None, optimizer, logger="wandb", batch_size=256, log_interval=10)

    solver.load(WANDB_PATH + '/model.pth')

    solver.train(num_epoch=20)
