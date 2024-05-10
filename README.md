# Bandit-based-HPO
Towards Bandit-based Optimization for Automated Machine Learning

This repository contains the implementation of the paper "Towards Bandit-based Optimization for Automated Machine Learning, accepted at ICLR 2024 Workshop on Practical Machine Learning for Low Resource Settings (PML4LRS) by Amir Rezaei Balef, Claire Vernade and, Katharina Eggensperger"


## Dependency

Using a Conda environment is recommended.

You may need to install and set up the TabRepo and YAHPO gym packages.

TabRepo: https://github.com/autogluon/tabrepo

YAHPO gym: https://github.com/slds-lmu/yahpo_gym



To install the repository, ensure you are using Python 3.9-3.11. Other Python versions may not be supported. Then, run the following commands:

```bash
git clone https://github.com/amirbalef/Bandit-based-HPO
pip install -r requirements.txt
```

Only Linux support has been tested.

## Running expiriments
To run experiments, execute the following command:

```bash
python main.py 
```

Feel free to adapt and extend this codebase as needed for your own experiments and research.