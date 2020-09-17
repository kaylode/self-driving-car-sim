# Self-driving car simulation using CNN (Pytorch)

Udacity simulator: https://github.com/udacity/self-driving-car-sim

## Training:
- Generate data using simulator
- Modify the model in models.regressor, output can be steering angle (and throttle)
- Run train.py

```
python train.py
```

## Inference:
- Start simulation in autonomous mode
- Load model checkpoint
- Run command:

```
python drive.py
```
## Results
![Alt Text](results/result.gif)
