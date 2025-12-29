import sys
print("Testing dependencies...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
except:
    print("❌ PyTorch not installed")
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except:
    print("❌ Transformers not installed")
try:
    import datasets
    print(f"✅ Datasets {datasets.__version__}")
except:
    print("❌ Datasets not installed")
