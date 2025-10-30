"""
Helper function for calling LLMs in a batch using multiprocessing
"""
import multiprocessing as mp
from typing import List

def batch_call_llm(llm_client: callable, prompts: List[str]) -> List[str]:
    with mp.Pool(processes=len(prompts)) as pool:
        results = pool.map(llm_client, prompts)
    return results