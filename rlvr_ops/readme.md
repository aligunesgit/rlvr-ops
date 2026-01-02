# RLVR-Ops: Production-Ready MLOps Framework for Reinforcement Learning with Verifiable Rewards

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-PeerJ%20CS-green.svg)](https://peerj.com)

RLVR-Ops is the first comprehensive MLOps framework specifically designed for Reinforcement Learning with Verifiable Rewards (RLVR). The framework provides production-ready infrastructure for training, deploying, and monitoring RLVR-based language models.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Quick Start](#quick-start)
- [Code Structure](#code-structure)
- [Usage Examples](#usage-examples)
- [Methodology](#methodology)
- [API Documentation](#api-documentation)
- [Training](#training)
- [Deployment](#deployment)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

## ✨ Features

- **GRPO Implementation**: Production-grade Group Relative Policy Optimization algorithm
- **Verifiable Reward Library**: 6 pre-implemented reward types (exact match, code execution, F1 score, classification, regression, custom)
- **Multi-Rollout Generation**: Generate k candidate responses with temperature control
- **Production Deployment**: FastAPI server, Docker containerization, Kubernetes manifests
- **Monitoring & Observability**: Training metrics, reward distributions, model performance tracking
- **Modular Architecture**: Easy to extend with custom policies, rewards, and training algorithms
- **Open Source**: MIT license, fully documented, community-driven

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (optional, for GPU support)

### Install from Source
```bash
# Clone repository
git clone https://github.com/aligunesgit/rlvr-ops.git
cd rlvr-ops

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies
```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

## 📊 Dataset Information

### GSM8k (Grade School Math 8K)

- **Source**: [OpenAI GSM8k](https://github.com/openai/grade-school-math)
- **Description**: Multi-step mathematical reasoning problems
- **Size**: 
  - Training: 7,473 problems
  - Test: 1,319 problems
- **Format**: Each problem contains a question and a step-by-step solution
- **License**: MIT
- **Task**: Generate numerical answers to grade-school math word problems
- **Evaluation**: Exact numerical match

### Data Preprocessing

The dataset undergoes minimal preprocessing:
1. Extract question text from JSON format
2. Format prompt: `"Question: {text}"`
3. Extract final numerical answer from solution
4. Normalize answer format (remove commas, spaces)

## ⚡ Quick Start

### Basic Usage
```python
from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.core.agent import RLVRAgent
from rlvr_ops.rewards.library import VerifiableRewardLibrary
from rlvr_ops.utils.data_loader import create_gsm8k_dataloader

# Initialize components
policy = GPT2Policy(model_name="gpt2")
agent = RLVRAgent(
    policy=policy,
    environment=None,
    reward_fn=VerifiableRewardLibrary.exact_match_reward
)

# Load data
train_loader = create_gsm8k_dataloader(split="train", batch_size=1, max_samples=50)

# Generate and evaluate
for batch in train_loader:
    prompt = batch['prompts'][0]
    ground_truth = batch['final_answers'][0]
    
    # Generate multiple rollouts
    rollouts = agent.generate_multi_rollout(
        prompt=prompt,
        ground_truth=ground_truth,
        num_rollouts=4,
        temperatures=[0.7, 0.8, 0.9, 1.0]
    )
    
    # Compute advantages
    rollouts = agent.compute_advantages(rollouts)
    
    # Select best
    best = agent.select_best_rollout(rollouts)
    print(f"Best reward: {best['reward']}")
```

## 📁 Code Structure
```
rlvr-ops/
├── rlvr_ops/                  # Main package
│   ├── core/                  # Core components
│   │   ├── policy.py          # Base policy interface
│   │   ├── gpt2_policy.py     # GPT-2 implementation
│   │   ├── agent.py           # RLVR agent
│   │   └── environment.py     # Environment interface
│   ├── training/              # Training components
│   │   ├── grpo.py            # GRPO algorithm
│   │   └── trainer.py         # Training engine
│   ├── rewards/               # Reward functions
│   │   ├── library.py         # Reward library
│   │   └── verifier.py        # Reward verification
│   ├── deployment/            # Deployment
│   │   ├── server.py          # FastAPI server
│   │   ├── Dockerfile         # Docker configuration
│   │   └── kubernetes/        # K8s manifests
│   ├── utils/                 # Utilities
│   │   ├── data_loader.py     # Data loading
│   │   └── metrics.py         # Metrics tracking
│   └── monitoring/            # Monitoring
│       └── logger.py          # Training logger
├── examples/                  # Example scripts
│   ├── train_gsm8k.py         # Training example
│   ├── evaluate.py            # Evaluation script
│   └── serve.py               # Server example
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── README.md                  # This file
└── LICENSE                    # MIT License
```

## 💡 Usage Examples

### Training on GSM8k
```python
from rlvr_ops.training.trainer import GRPOTrainer
from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.core.agent import RLVRAgent
from rlvr_ops.rewards.library import VerifiableRewardLibrary
from rlvr_ops.utils.data_loader import create_gsm8k_dataloader

# Initialize
policy = GPT2Policy(model_name="gpt2")
agent = RLVRAgent(
    policy=policy,
    environment=None,
    reward_fn=VerifiableRewardLibrary.exact_match_reward
)

# Create data loaders
train_loader = create_gsm8k_dataloader(split="train", batch_size=1, max_samples=100)
val_loader = create_gsm8k_dataloader(split="test", batch_size=1, max_samples=30)

# Initialize trainer
trainer = GRPOTrainer(
    policy=policy,
    agent=agent,
    learning_rate=5e-6,
    num_rollouts=4,
    temperatures=[0.7, 0.8, 0.9, 1.0]
)

# Train
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=3,
    save_dir="./checkpoints"
)
```

### Command Line Training
```bash
python examples/train_gsm8k.py \
    --model gpt2 \
    --epochs 3 \
    --rollouts 4 \
    --learning-rate 5e-6 \
    --max-samples 100 \
    --save-dir ./checkpoints
```

### Custom Reward Function
```python
from rlvr_ops.rewards.library import VerifiableRewardLibrary

@VerifiableRewardLibrary.custom_reward
def my_custom_reward(prediction: str, ground_truth: str) -> float:
    """Custom domain-specific reward function."""
    # Your logic here
    score = compute_similarity(prediction, ground_truth)
    return float(score)  # Must return value in [0, 1]

# Use in agent
agent = RLVRAgent(
    policy=policy,
    environment=None,
    reward_fn=my_custom_reward
)
```

## 🔬 Methodology

### GRPO Algorithm

RLVR-Ops implements Group Relative Policy Optimization (GRPO):

1. **Generate k rollouts** per input with varying temperatures
2. **Compute verifiable rewards** for each rollout
3. **Calculate baseline** as mean reward across rollouts
4. **Compute advantages**: A_i = R_i - baseline
5. **Update policy** using advantage-weighted policy gradient

### Assessment Metrics

- **Accuracy**: Percentage of problems where at least one of k rollouts produces the correct answer
- **Mean Reward**: Average reward across all rollouts
- **Best-of-k Reward**: Maximum reward among k rollouts per input
- **Training Loss**: Policy gradient loss with advantage weighting

**Justification**: Accuracy is appropriate for mathematical reasoning as answers are objectively verifiable. Best-of-k accuracy evaluates the framework's multi-rollout selection capability.

## 🌐 API Documentation

### Start Server
```bash
# Development
python -m rlvr_ops.deployment.server

# Production with Uvicorn
uvicorn rlvr_ops.deployment.server:app --host 0.0.0.0 --port 8000
```

### API Endpoints

**POST /generate**

Generate responses with multi-rollout RLVR.
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 15 + 27?",
    "num_rollouts": 4,
    "temperature": 0.9,
    "ground_truth": "42"
  }'
```

Response:
```json
{
  "rollouts": [
    {"response": "42", "reward": 1.0, "temperature": 0.9},
    {"response": "42", "reward": 1.0, "temperature": 0.9},
    {"response": "43", "reward": 0.0, "temperature": 0.9},
    {"response": "41", "reward": 0.0, "temperature": 0.9}
  ],
  "best_response": "42",
  "best_reward": 1.0,
  "mean_reward": 0.5
}
```

**GET /health**

Check server health.
```bash
curl http://localhost:8000/health
```

**Interactive API Docs**: Visit `http://localhost:8000/docs`

## 🐳 Deployment

### Docker
```bash
# Build image
docker build -t rlvr-ops:latest .

# Run container
docker run -p 8000:8000 rlvr-ops:latest

# With GPU support
docker run --gpus all -p 8000:8000 rlvr-ops:latest
```

### Kubernetes
```bash
# Deploy to K8s
kubectl apply -f rlvr_ops/deployment/kubernetes/

# Check status
kubectl get pods -l app=rlvr-ops

# Access service
kubectl port-forward svc/rlvr-ops-service 8000:8000
```

## 📝 Citation

If you use RLVR-Ops in your research, please cite:
```bibtex
@article{gunes2025rlvrops,
  title={RLVR-Ops: A Production-Ready MLOps Framework for Reinforcement Learning with Verifiable Rewards},
  author={Gunes, Ali},
  journal={PeerJ Computer Science},
  year={2025},
  url={https://github.com/aligunesgit/rlvr-ops}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/aligunesgit/rlvr-ops.git
cd rlvr-ops
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black rlvr_ops/
flake8 rlvr_ops/
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

**Ali Gunes**
- Email: your.email@institution.edu
- GitHub: [@aligunesgit](https://github.com/aligunesgit)
- Project: [https://github.com/aligunesgit/rlvr-ops](https://github.com/aligunesgit/rlvr-ops)

## 🙏 Acknowledgments

- OpenAI for the [GSM8k dataset](https://github.com/openai/grade-school-math)
- HuggingFace for [Transformers library](https://github.com/huggingface/transformers)
- PyTorch team for [PyTorch framework](https://pytorch.org/)

## 🔗 Related Work

- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **DeepSeek-R1**: [DeepSeek-R1 Technical Report](https://github.com/deepseek-ai/DeepSeek-R1)
- **GSM8k Dataset**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

---

**Made with ❤️ for the RLVR community**