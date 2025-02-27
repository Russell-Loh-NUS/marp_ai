# marp_ai
This is a machine learning based Multi Agent Route Planning (MARP) project for NUS MSc in Robotics module CS5446: AI Planning and Decision Making.

This project aims to solve conflict between multiple AMR in a graph based fleet manager

## Setup
### 1. Clone the repo
```
git clone https://github.com/PiusLim373/marp_ai.git
```
### 2. Setup using Conda
```
cd marp_ai
conda env create -f requirement.yml
conda activate cs5446-marp-ai
```
This will create a conda environment cs5446-marp-ai with necessary packages installed to run the project.

## Training with Stable Baselines3 PPO Algorithm
```
python training_agent.py
```
:warning: User can opt to render the matplotlib visualizer during training, but the training time will be significantly slower
![](/docs/visualizer.png)

After the training completed, a `ppo_marp_ai_model.zip` will be saved to the same directory

## Testing with Trained Model
```
python testing_agent.py
```
