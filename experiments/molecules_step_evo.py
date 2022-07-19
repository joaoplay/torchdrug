from torch import optim

from torchdrug import datasets, models, tasks
from torchdrug.core import Engine

if __name__ == '__main__':
    dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True, node_feature="symbol")
    model = models.RGCN(input_dim=dataset.node_feature_dim,
                        num_relation=dataset.num_bond_type,
                        hidden_dims=[256, 256, 256, 256], batch_norm=False)
    task = tasks.GCPNGeneration(model, dataset.atom_types, max_edge_unroll=12,
                                max_node=38, criterion="nll")
    optimizer = optim.Adam(task.parameters(), lr=1e-3)
    solver = Engine(task, dataset, None, None, optimizer, batch_size=128, log_interval=10)
    solver.train(num_epoch=1)
