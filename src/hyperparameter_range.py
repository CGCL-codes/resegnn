hp_range_ICEWS14 = {
    "history_len": [9],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [12],
    "sem_rate": [0.3],
    "task_weight": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    # "history_method": ["once", "time_decay"]
}

# hp_range_WIKI = {
#     "history_len": [1, 2,3],
#     "dropout": [0.2],
#     "n_bases": [100],
#     "angle": [10],
#     "sem_rate": [0.3,0.5],
#     "task_weight": [0.5, 0.6, 0.7],
#     # "history_method": ["once", "time_decay"]
# }

hp_range_WIKI = {
    "history_len": [1, 2,3],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [10],
    "sem_rate": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    "task_weight": [0.6],
    # "history_method": ["once", "time_decay"]
}

hp_range_YAGO = {
    "history_len": [1],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [10],
    "sem_rate": [0.5],
    "task_weight": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    # "history_method": ["once", "time_decay"]
}

hp_range_ICEWS18 = {
    "history_len": [10],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [10, 12],
    "sem_rate": [0.3,0.5],
    "task_weight": [0.6, 0.7],
    # "history_method": ["once", "time_decay"]
}

hp_range_ICEWS05_15 = {
    "history_len": [15,14,16],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [10, 12],
    "sem_rate": [0.3,0.4],
    "task_weight": [0.6, 0.7],
    # "history_method": ["once", "time_decay"]
}

hp_range_GDELT = {
    "history_len": [6, 7],
    "dropout": [0.2],
    "n_bases": [100],
    "angle": [10],
    "sem_rate": [0.3,0.5],
    "task_weight": [0.6, 0.7],
    # "history_method": ["once", "time_decay"]
}
