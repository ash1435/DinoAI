# DinoAI

An AI built to run the ever popular Dino game from Chrome. A simple AI constructed using DQNs and implemented using pytorch.
Keep in mind it is compuationally intensive and may crash your system if run for long periods.

NOTE: It is not perfect and will not work as you expect it to.

## Tesseract installer for Windows

Normally we run Tesseract on Debian GNU Linux, but there was also the need for a Windows version. That's why we have built a Tesseract installer for Windows.

WARNING: Tesseract should be either installed in the directory which is suggested during the installation or in a new directory. The uninstaller removes the whole installation directory. If you installed Tesseract in an existing directory, that directory will be removed with all its subdirectories and files.

The latest installers can be downloaded here:

* [tesseract-ocr-w32-setup-v5.2.0.20220712.exe](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w32-setup-v5.2.0.20220712.exe) (32 bit) 
* [tesseract-ocr-w64-setup-v5.2.0.20220712.exe](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.2.0.20220712.exe) (64 bit) resp.

## Install dependencies

Pytorch is not supported on Python 3.11 hence I recommend using a virtual env.

Install miniconda from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe).

Open up the anaconda terminal from the start menu and run the following commands:

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
## Run AI

```bash
python main.py [number_of_episodes]
```
The dafault number of episodes is 1.
