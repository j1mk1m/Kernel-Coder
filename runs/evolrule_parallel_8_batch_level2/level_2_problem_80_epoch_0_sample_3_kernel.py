import torch
        from torch.utils.benchmark import Timer

        model = Model(*get_init_inputs())
        inputs = get_inputs()
        t = Timer(stmt='model(*inputs)', globals=globals(), num_threads=1)
        print(t.blocked_autorange())

        model_new = ModelNew(*get_init_inputs())
        inputs = get_inputs()
        t = Timer(stmt='model_new(*inputs)', globals=globals(), num_threads=1)
        print(t.blocked_autorange())