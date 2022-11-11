# DinoAI

An AI built to run the ever popular Dino game from Chrome. A simple AI constructed using DQNs and implemented using pytorch.

### Install dependencies

Pytorch is not supported on Python 3.11 hence I recommend using a virtual env.

Install miniconda from https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe.

Open up the anaconda terminal and run the following commands:

```bash
conda create -n [name_of_your_venv] pip
```
```bash
conda activate [name_of_your_venv]
```
Navigate to the DinoAI folder and execute the rest

```bash
pip install -r requirements.txt
```
### Run AI

```bash
python main.py [number_of_episodes]
```
The dafault number of episodes is 1.
