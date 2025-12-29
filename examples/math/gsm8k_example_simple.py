def exact_match_reward(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

def example_gsm8k():
    print("=" * 60)
    print("GSM8k Math Problem Example")
    print("=" * 60)
    
    problem = "Janet has 3 apples and buys 2 more. How many apples does she have?"
    correct_answer = "5"
    model_prediction = "5"
    
    reward = exact_match_reward(model_prediction, correct_answer)
    print(f"\nProblem: {problem}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Model Prediction: {model_prediction}")
    print(f"Reward: {reward}")
    print(f"\n✅ RLVR Framework is working!")
    
if __name__ == '__main__':
    example_gsm8k()
