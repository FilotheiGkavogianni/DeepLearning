*Train a model :*
> train a baseline MLP model
```
python main.py --model "mlp" --task "train"
```
> train an MLP model with tuning
```
python main.py --model "mlp" --tune --task "train"
```
> train a baseline CNN model
```
python main.py --model "cnn" --task "train"
```
> train an CNN model with tuning
```
python main.py --model "cnn" --tune --task "train"
```


*Predict on an already trained model:*
> predict using a baseline MLP model
```
python main.py --model "mlp" --task "test"
```
> predict using a tuned MLP model
```
python main.py --model "mlp" --tune --task "test"
```
> predict using a baseline CNN model
```
python main.py --model "cnn" --task "test"
```
> predict using a tuned CNN model
```
python main.py --model "cnn" --tune --task "test"
```
