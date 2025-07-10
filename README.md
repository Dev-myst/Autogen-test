# Autogen-Conference


## Setup Ollama
Install Ollama by following the instructions on the official website for your operating system. Link for download: https://ollama.com/ 

## Download Models
Use the following commands to download the required models:
 ```bash
   
ollama pull granite3.3:2b
ollama pull granite3.3:8b
```

## Setup Instructions - Run locally

1. **Create an virtual enviroment**
   ```bash
   pip install virtualenv
   virtualenv autogen_env

   source autogen_env/bin/activate

2. **Install Autogen**
   ```bash
   pip install pip install -U "autogen-agentchat"
   pip install -U "autogen-ext[ollama]"
   pip install streamlit
   
   
3. **Run locally - interface**
   ```bash
   streamlit run main.py
