import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rlvr_ops.rewards.library import VerifiableRewardLibrary

def example_gsm8k():
    print("=" * 60)
    print("GSM8k Math Problem Example")
    print("=" * 60)
    
    problem = "Janet has 3 apples and buys 2 more. How many apples does she have?"
    correct_answer = "5"
    model_prediction = "5"
    
    reward = VerifiableRewardLibrary.exact_match_reward(model_prediction, correct_answer)
    print(f"\nProblem: {problem}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Model Prediction: {model_prediction}")
    print(f"Reward: {reward}")
    
if __name__ == '__main__':
    example_gsm8k()
