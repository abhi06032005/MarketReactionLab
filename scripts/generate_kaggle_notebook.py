import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MarketReactionLab: GPU Fine-Tuning Pipeline on Kaggle\n",
    "\n",
    "**Project:** MarketReactionLab  \n",
    "**Objective:** End-to-end GPU fine-tuning of FinBERT + Numerical Market Indicators for Short-Term Post-Earnings Equity Reaction Prediction.  \n",
    "**Hardware:** Kaggle GPU (NVIDIA T4 x2 or P100) with PyTorch and Hugging Face Transformers."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Environment Setup and Repository Clone"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone the clean research repository\n",
    "!git clone https://github.com/abhi06032005/MarketReactionLab.git\n",
    "%cd MarketReactionLab\n",
    "\n",
    "# Install dependencies\n",
    "!pip install -q transformers accelerate scikit-learn pyarrow\n",
    "\n",
    "import torch\n",
    "print(f\"PyTorch Version: {torch.__version__}\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"GPU Device: {torch.cuda.get_device_name(0)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Dataset Preparation and Inspection\n",
    "\n",
    "Run Phase 1 data pipeline to generate the canonical chronological splits (`train.parquet`, `validation.parquet`, `test.parquet`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.data.pipeline import run_pipeline\n",
    "import pandas as pd\n",
    "import os\n",
    "\n",
    "# Check if processed parquets exist, otherwise run pipeline\n",
    "if not os.path.exists(\"data/processed/train.parquet\"):\n",
    "    print(\"Running data processing pipeline...\")\n",
    "    run_pipeline()\n",
    "\n",
    "train_df = pd.read_parquet(\"data/processed/train.parquet\")\n",
    "val_df = pd.read_parquet(\"data/processed/validation.parquet\")\n",
    "test_df = pd.read_parquet(\"data/processed/test.parquet\")\n",
    "\n",
    "print(f\"Train samples: {len(train_df):,} (Date range: {train_df['event_date'].min()} to {train_df['event_date'].max()})\")\n",
    "print(f\"Val samples:   {len(val_df):,} (Date range: {val_df['event_date'].min()} to {val_df['event_date'].max()})\")\n",
    "print(f\"Test samples:  {len(test_df):,} (Date range: {test_df['event_date'].min()} to {test_df['event_date'].max()})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Dataset Tokenization and Collation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from transformers import AutoTokenizer\n",
    "from src.training.dataset import FinancialReactionDataset\n",
    "from torch.utils.data import DataLoader\n",
    "from src.training.collate import multimodal_collate_fn\n",
    "\n",
    "MODEL_NAME = \"ProsusAI/finbert\"\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
    "\n",
    "train_dataset = FinancialReactionDataset(train_df, tokenizer, max_length=128)\n",
    "val_dataset = FinancialReactionDataset(val_df, tokenizer, max_length=128)\n",
    "test_dataset = FinancialReactionDataset(test_df, tokenizer, max_length=128)\n",
    "\n",
    "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=multimodal_collate_fn)\n",
    "val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
    "test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=multimodal_collate_fn)\n",
    "\n",
    "print(\"DataLoaders prepared successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Multimodal Fusion Model Architecture and Optimizer Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.models.multimodal_fusion import MultimodalFusionModel\n",
    "import torch.nn as nn\n",
    "from torch.optim import AdamW\n",
    "\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "print(f\"Using compute device: {device}\")\n",
    "\n",
    "model = MultimodalFusionModel(\n",
    "    pretrained_model=MODEL_NAME,\n",
    "    num_features=19,\n",
    "    num_classes=3,\n",
    "    freeze_layers=10  # Unfreeze layers 10 and 11 for fine-tuning\n",
    ").to(device)\n",
    "\n",
    "# Set discriminative learning rates\n",
    "finbert_params = [p for n, p in model.bert.named_parameters() if p.requires_grad]\n",
    "fusion_params = [p for n, p in model.named_parameters() if not n.startswith(\"bert\")]\n",
    "\n",
    "optimizer = AdamW([\n",
    "    {\"params\": finbert_params, \"lr\": 2e-5, \"weight_decay\": 0.01},\n",
    "    {\"params\": fusion_params, \"lr\": 1e-3, \"weight_decay\": 0.01}\n",
    "])\n",
    "\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))\n",
    "print(\"Model and optimizer initialized.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. GPU Training Loop with Early Stopping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from pathlib import Path\n",
    "from tqdm.auto import tqdm\n",
    "\n",
    "EPOCHS = 5\n",
    "best_val_loss = float('inf')\n",
    "checkpoint_dir = Path(\"models/checkpoints\")\n",
    "checkpoint_dir.mkdir(parents=True, exist_ok=True)\n",
    "best_model_path = checkpoint_dir / \"best_finetuned_fusion.pt\"\n",
    "\n",
    "for epoch in range(1, EPOCHS + 1):\n",
    "    model.train()\n",
    "    total_loss, correct, total = 0.0, 0, 0\n",
    "    pbar = tqdm(train_loader, desc=f\"Epoch {epoch}/{EPOCHS} [Train]\")\n",
    "    \n",
    "    for batch in pbar:\n",
    "        optimizer.zero_grad()\n",
    "        input_ids = batch['input_ids'].to(device)\n",
    "        attention_mask = batch['attention_mask'].to(device)\n",
    "        features = batch['features'].to(device)\n",
    "        labels = batch['label'].to(device)\n",
    "        \n",
    "        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):\n",
    "            logits = model(input_ids, attention_mask, features)\n",
    "            loss = criterion(logits, labels)\n",
    "            \n",
    "        scaler.scale(loss).backward()\n",
    "        scaler.unscale_(optimizer)\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n",
    "        scaler.step(optimizer)\n",
    "        scaler.update()\n",
    "        \n",
    "        total_loss += loss.item() * len(labels)\n",
    "        preds = logits.argmax(dim=-1)\n",
    "        correct += (preds == labels).sum().item()\n",
    "        total += len(labels)\n",
    "        pbar.set_postfix({\"loss\": f\"{total_loss/total:.4f}\", \"acc\": f\"{correct/total:.4f}\"})\n",
    "        \n",
    "    # Validation Loop\n",
    "    model.eval()\n",
    "    val_loss, val_correct, val_total = 0.0, 0, 0\n",
    "    with torch.no_grad():\n",
    "        for batch in val_loader:\n",
    "            input_ids = batch['input_ids'].to(device)\n",
    "            attention_mask = batch['attention_mask'].to(device)\n",
    "            features = batch['features'].to(device)\n",
    "            labels = batch['label'].to(device)\n",
    "            \n",
    "            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):\n",
    "                logits = model(input_ids, attention_mask, features)\n",
    "                loss = criterion(logits, labels)\n",
    "                \n",
    "            val_loss += loss.item() * len(labels)\n",
    "            preds = logits.argmax(dim=-1)\n",
    "            val_correct += (preds == labels).sum().item()\n",
    "            val_total += len(labels)\n",
    "            \n",
    "    epoch_val_loss = val_loss / val_total\n",
    "    epoch_val_acc = val_correct / val_total\n",
    "    print(f\"Epoch {epoch} Summary: Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}\")\n",
    "    \n",
    "    if epoch_val_loss < best_val_loss:\n",
    "        best_val_loss = epoch_val_loss\n",
    "        torch.save(model.state_dict(), best_model_path)\n",
    "        print(f\"Saved best checkpoint to {best_model_path}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Full Test Set Evaluation and Trading Simulation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report, matthews_corrcoef\n",
    "import numpy as np\n",
    "\n",
    "# Load best checkpoint\n",
    "model.load_state_dict(torch.load(best_model_path, map_location=device))\n",
    "model.eval()\n",
    "\n",
    "all_preds, all_labels = [], []\n",
    "with torch.no_grad():\n",
    "    for batch in test_loader:\n",
    "        input_ids = batch['input_ids'].to(device)\n",
    "        attention_mask = batch['attention_mask'].to(device)\n",
    "        features = batch['features'].to(device)\n",
    "        labels = batch['label'].to(device)\n",
    "        \n",
    "        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):\n",
    "            logits = model(input_ids, attention_mask, features)\n",
    "            preds = logits.argmax(dim=-1)\n",
    "            \n",
    "        all_preds.extend(preds.cpu().numpy())\n",
    "        all_labels.extend(labels.cpu().numpy())\n",
    "\n",
    "all_preds = np.array(all_preds)\n",
    "all_labels = np.array(all_labels)\n",
    "\n",
    "label_names = [\"Negative (-1)\", \"Neutral (0)\", \"Positive (+1)\"]\n",
    "print(\"=== Full Test Set Classification Report ===\")\n",
    "print(classification_report(all_labels, all_preds, target_names=label_names, digits=4))\n",
    "print(f\"Matthews Correlation Coefficient (MCC): {matthews_corrcoef(all_labels, all_preds):.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Institutional Backtest with Calendar Aggregation and 10 bps Costs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Map 0, 1, 2 predictions to -1, 0, +1 trading actions\n",
    "idx_to_label = {0: -1, 1: 0, 2: 1}\n",
    "pred_actions = np.array([idx_to_label[p] for p in all_preds])\n",
    "\n",
    "sim_df = test_df.copy()\n",
    "sim_df['pred_action'] = pred_actions\n",
    "sim_df['trade_return'] = sim_df['pred_action'] * sim_df['excess_return'] - (sim_df['pred_action'] != 0) * 0.0010\n",
    "\n",
    "# Daily calendar aggregation\n",
    "daily_returns = sim_df.groupby('event_date')['trade_return'].mean()\n",
    "\n",
    "mean_daily = daily_returns.mean()\n",
    "std_daily = daily_returns.std()\n",
    "annualized_sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0.0\n",
    "cumulative_return = (1 + daily_returns).prod() - 1\n",
    "\n",
    "print(\"=== Backtest Performance ===\")\n",
    "print(f\"Total Test Days:        {len(daily_returns)}\")\n",
    "print(f\"Mean Daily Return:      {mean_daily*100:.3f}%\")\n",
    "print(f\"Daily Volatility:       {std_daily*100:.3f}%\")\n",
    "print(f\"Annualized Sharpe:      {annualized_sharpe:.3f}\")\n",
    "print(f\"Cumulative Return:      {cumulative_return*100:.1f}%\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbformat": 4,
   "nbformat_minor": 2
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = Path("notebooks/kaggle_marketreactionlab_training.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {output_path} successfully.")
