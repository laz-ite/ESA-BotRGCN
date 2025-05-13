ESA-BotRGCN: GNN-Based Social Bot Detection on Twibot-20
=========================================================

Project Overview:
-----------------
This project implements ESA-BotRGCN, a Graph Neural Network (GNN)-based framework for social bot detection using the Twibot-20 dataset. It integrates user textual content, profile features, and social graph structure to classify users as either bots or humans.

Supported models include:
- ESA-BotRGCN
- ESA-FastBotRGCN
- ESA-BotGAT
- ESA-BotRGAT
- ESA-BotGCN
- ESA-BotMLP
- ESA-BotRGCNWithAttention

Directory Structure:
--------------------
- model/               GNN model definitions (e.g., ESA-BotRGCN variants)
- Dataset.py           Dataset loading and preprocessing pipeline
- utils.py             Utility functions (accuracy, weight initialization, etc.)
- main.py              Main training and evaluation script
- Data/                Directory for intermediate tensor and graph data
- README.txt           Project documentation

Dependencies:
-------------
- Python 3.8+
- PyTorch
- torch_geometric
- transformers
- numpy
- pandas
- tqdm
- scikit-learn

Installation:
-------------
1. (Optional) Create a virtual environment:
   conda create -n twibot python=3.8
   conda activate twibot

2. Install dependencies:
   pip install -r requirements.txt

3. Download and place the Twibot-20 dataset JSON files into the project root:
   - train.json
   - dev.json
   - test.json
   - support.json

Usage:
------
To train and evaluate ESA-BotRGCN:
   python main.py

To switch models, modify `main.py` to use:
   - ESA-BotRGCN
   - ESA-FastBotRGCN
   - ESA-BotGAT
   - ESA-BotRGAT
   - ESA-BotGCN
   - ESA-BotMLP
   - ESA-BotRGCNWithAttention

Evaluation Metrics:
-------------------
- Accuracy
- F1 Score
- Matthews Correlation Coefficient (MCC)

Highlights:
-----------
- Utilizes RoBERTa for embedding descriptions and tweets.
- Combines categorical, numerical, and sentiment features.
- Features and graph edges are cached in `Data/` for efficiency.
- Compatible with GPU acceleration.

License:
--------
This project is released under the MIT License.

Contact:
--------
For questions, please open an issue on GitHub or contact the maintainer directly.
