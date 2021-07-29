from importlib import import_module

from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())   ## load the right dataset loader module
            trainset = getattr(module_train, args.data_train)(args)   ### load the dataset, args.data_train is the  dataset name
            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100', 'DIV2K']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, name=args.data_test,train=False)
        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False)

        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=not args.cpu
        )
