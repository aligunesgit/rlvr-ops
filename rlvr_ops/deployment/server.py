"""
FastAPI Production Server for RLVR-Ops
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlvr_ops.core.gpt2_policy import GPT2Policy
from rlvr_ops.rewards.library import VerifiableRewardLibrary
from rlvr_ops.core.agent import RLVRAgent

# Initialize FastAPI
app = FastAPI(
    title="RLVR-Ops API",
    description="Production API for RLVR model serving",
    version="1.0.0"
)

# Global model (loaded on startup)
policy = None
agent = None


class GenerateRequest(BaseModel):
    """Request model for generation endpoint."""
    prompt: str = Field(..., description="Input prompt for generation")
    num_rollouts: int = Field(4, ge=1, le=16, description="Number of rollouts")
    max_tokens: int = Field(100, ge=1, le=512, description="Max tokens to generate")
    temperature: float = Field(0.9, ge=0.1, le=2.0, description="Sampling temperature")
    ground_truth: Optional[str] = Field(None, description="Ground truth for reward computation")


class GenerateResponse(BaseModel):
    """Response model for generation endpoint."""
    rollouts: List[dict]
    best_response: str
    best_reward: float
    mean_reward: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global policy, agent
    
    print("Loading GPT-2 model...")
    policy = GPT2Policy(model_name="gpt2")
    
    # Initialize agent with exact match reward
    agent = RLVRAgent(
        policy=policy,
        environment=None,  # Not needed for inference
        reward_fn=VerifiableRewardLibrary.exact_match_reward
    )
    
    print("✅ Model loaded successfully!")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=policy is not None,
        device=str(policy.device) if policy else "unknown"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate responses with multi-rollout RLVR.
    
    Args:
        request: Generation request parameters
        
    Returns:
        Generated rollouts with rewards
    """
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate rollouts
        temperatures = [request.temperature] * request.num_rollouts
        
        rollouts = agent.generate_multi_rollout(
            prompt=request.prompt,
            ground_truth=request.ground_truth or "",
            num_rollouts=request.num_rollouts,
            temperatures=temperatures,
            max_new_tokens=request.max_tokens
        )
        
        # Find best
        best = agent.select_best_rollout(rollouts)
        
        # Compute mean reward
        mean_reward = sum(r['reward'] for r in rollouts) / len(rollouts)
        
        return GenerateResponse(
            rollouts=rollouts,
            best_response=best['response'],
            best_reward=best['reward'],
            mean_reward=mean_reward
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate")
async def evaluate(
    response: str,
    ground_truth: str,
    reward_type: str = "exact_match"
):
    """
    Evaluate a response against ground truth.
    
    Args:
        response: Generated response
        ground_truth: Expected answer
        reward_type: Type of reward function
        
    Returns:
        Reward score
    """
    reward_functions = {
        "exact_match": VerifiableRewardLibrary.exact_match_reward,
        "classification": VerifiableRewardLibrary.classification_reward,
        "regression": VerifiableRewardLibrary.regression_reward,
    }
    
    if reward_type not in reward_functions:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown reward type: {reward_type}"
        )
    
    reward_fn = reward_functions[reward_type]
    reward = reward_fn(response, ground_truth)
    
    return {
        "reward": reward,
        "reward_type": reward_type
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RLVR-Ops API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/generate": "Generate with multi-rollout RLVR",
            "/evaluate": "Evaluate response with reward function",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )