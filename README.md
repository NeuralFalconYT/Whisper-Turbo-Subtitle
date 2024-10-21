# Auto Subtitle Generator Using Whisper-Large-V3-Turbo-Ct2
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/Whisper-Turbo-Subtitle/blob/main/Whisper_Turbo_Subtitle.ipynb) <br>
[![hfspace](https://img.shields.io/badge/🤗-Space%20demo-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS) <br>

### Step 1:
```
git clone https://github.com/NeuralFalconYT/Whisper-Turbo-Subtitle.git
```
### Step 2:
```
cd Whisper-Turbo-Subtitle
```
### Step 3:
```
pip install -r requirements.txt
```
You may need to install [ffmpeg](https://www.ffmpeg.org/download.html) <br>
### Step 4:
For local development:
```
python app.py 
```
For debugging:
```
python app.py --debug 
```
For cloud servers or notebooks:
```
python app.py --debug --share
```
```--debug```: Enables debug mode, providing more detailed logs and helpful error messages.<br>
```--share```: Shares the application by generating a public link, allowing access from external networks.
