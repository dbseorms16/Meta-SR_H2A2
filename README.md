2. run training demo:
```
python main.py --model metardn --ext sep  --save metardn --lr_decay 200 --epochs 1000 --n_GPUs 1 --batch_size 1
```

python main.py --model rdn --save rdn --ext sep --lr_decay 200 --epochs 1000 --n_GPUs 1 --batch_size 1 --scale 2.0 --pre_train ./experiment/rdn/model/model_best.pt --save_results