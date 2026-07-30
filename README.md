# Metabolic Bacterial Game
Bacterial communities exhibit a range of complex interactions, from competition to cooperation, shaped by access to shared environmental resources. 
To model these dynamics, we introduce Metabolic Bacterial Game (MetabGame) --- a reinforcement learning framework where bacteria strategically consume or produce environmental compounds based on their metabolic needs.   
![Logo](https://github.com/powidla/MetabGame/blob/main/assets/main.png)  

The game may operate in a cooperative or competitive mode, with rewards structured to reflect alignment or misalignment with the environmental context.

N-FBA implementation follows the [Friend-or-Foe repo](https://github.com/powidla/Friend-Or-Foe/tree/main/models/Matlab).

## Repository Structure
```
MetaBGame/
├── metabgame/                          # Core library
│   ├── ac.py             # Actor-Critic implementation
│   ├── config.py         # Config contains main parameters of the game
│   ├── env.py            # Main environment
│   ├── logs.py           # Logging of rewards, actions, aggregation for N agents 
│   ├── plots.py          # Plotting the rewards and behavior curves
│   ├── prime.py          # Random keys
│   ├── run.py            # Main simulation
│   ├── train.py          # training with PPO for Nash and Bayes
│   └── utils.py          # Utility functions: loading and summary
├── examples/
│   ├── agent.json        # Example of the agent structure
│   └── model.json        # Example of the model structure 
├── assets/
│   ├── main.png          # Overview figure in the repo
└── requirements/
```

## Reproducing main results
Set up the environment and download required packages
```bash
conda create -n mgame python=3.11
pip install -r requirements.txt
```
Run scaling experiments
```bash
python run.py
```
  
