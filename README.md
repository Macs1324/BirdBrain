# **BirdBrain**
A handmade AI experiment

![til](./demo.gif)

## What is this?

*A flappy bird clone controlled by a Deep neural network entirely built from python lists*

## Run it

Setting up the environment:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run of the following files using `python <filename>`

- `train_bird.py`: Run this file to start training the birds. After each generation the best brain gets saved to the auto_bestbrain.qb file WARNING: Extremely low performance, python lists are not for the faint-hearted
- `flappy_bird_clear.py`: Run this file to see how the best bird of the bunch (whose brain was saved in auto_bestbrain.qb) gracefully soars trough the sky




## QuickMaths and quickbrain
Two libraries I built long ago to attempt to challenge god at his own game with the least amount of pre-made tools possible. 
These libraries do have things like numpy and matplotlib as dependencies, but these deps have no critical role, and only serve for compatibility purposes

### q_learning
`An extra experiment at Q learning built on QuickMaths (and quickbrain). An agent in a grid-based enviroment (rendered in the console trough a qm Matrix) tries to find its way around the obstacles to get to the reward`
