import torch
from multiprocessing.spawn import freeze_support

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if checkpoint.ok:
    loader = data.Data(args)                ##data loader
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    def main():
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

    if __name__ == '__main__':  # 중복 방지를 위한 사용
        freeze_support()  # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만들어준다.
        main()