from dataclasses import dataclass
from typing import List, Tuple
import logging
import os
import json
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult
from llm_utils import create_llm_client


class KnowledgeBase:
    def __init__(self):
        pass

    def retrieve(self, query):
        pass

    def extract(self, trajectories: dict[Task, List[str]]) -> None:
        pass


@dataclass
class MemoryItem:
    title: str
    description: str
    content: str

class Memory(KnowledgeBase):
    def __init__(self, config, run_dir: str):
        self.run_dir = run_dir
        self.memory = []
        self.embeddings = []
        self.memory_path = os.path.join(self.run_dir, "memory.json")
        self.embeddings_path = os.path.join(self.run_dir, "embeddings.json")
        self.load_memory()
        self.config = config
        self.llm_client = create_llm_client(data_file=os.path.join(self.run_dir, "memory_llm_usage.json"), default_model=config.model_name, default_temperature=1.0)
        self.embedding_model = create_llm_client(data_file=os.path.join(self.run_dir, "memory_embedding_usage.json"), default_model="gemini/gemini-embedding-001")
    
    def _add_to_memory(self, memory_items: List[MemoryItem]) -> None:
        for memory_item in memory_items:
            self.memory.append(memory_item)
            self.embeddings.append(self._embed(memory_item))
    
    def save_memory(self) -> None:
        with open(self.memory_path, "w") as f:
            json.dump([self._memory_to_json(memory) for memory in self.memory], f, indent=2)
        with open(self.embeddings_path, "w") as f:
            json.dump(self.embeddings, f, indent=2)
        self.llm_client.save_usage_data()
        self.embedding_model.save_usage_data()
    
    def load_memory(self) -> None:
        if os.path.exists(self.memory_path) and os.path.exists(self.embeddings_path):
            with open(self.memory_path, "r") as f:
                self.memory = [self._json_to_memory(memory) for memory in json.load(f)]
            with open(self.embeddings_path, "r") as f:
                self.embeddings = json.load(f)
    
    def _embed(self, memory_item: MemoryItem) -> None:
        return self.embedding_model.embedding(self._memory_to_string(memory_item))["data"][0]["embedding"]

    def _memory_to_string(self, memory_item: MemoryItem) -> str:
        return f"## Title {memory_item.title}\n## Description {memory_item.description}\n## Content {memory_item.content}"
    
    def _memory_to_json(self, memory_item: MemoryItem) -> dict:
        return {
            "title": memory_item.title,
            "description": memory_item.description,
            "content": memory_item.content
        }
    
    def _json_to_memory(self, json: dict) -> MemoryItem:
        return MemoryItem(title=json["title"], description=json["description"], content=json["content"])

    def retrieve(self, query):
        # for each memory, calculate similarity and output top k most similar (default k=1)
        # gemini-embedding-001 model was used
        logger.info(f"Retrieving memory")
        if len(self.memory) == 0:
            return ""
        query_embedding = self.embedding_model.embedding(query)["data"][0]["embedding"]
        similarities = [np.dot(query_embedding, embedding) for embedding in self.embeddings]
        relevant_memory = [self.memory[np.argsort(similarities)[-1]]] # TODO: change to top k, top 1 for now
        answer = "Consider the following tips:" + "\n".join([self._memory_to_string(memory) for memory in relevant_memory])
        return answer

    def extract(self, epoch: int, trajectories: dict[Task, List[str]]) -> None:
        logger.info(f"Extracting memory")
        assert len(trajectories) == 1, "Memory extraction only supports a single task at a time"
        task, solutions = list(trajectories.items())[0]
        memory_path = os.path.join(self.run_dir, f"{task.task_id}_epoch_{epoch}_memory.json")
        if os.path.exists(memory_path):
            print(f"Loading memory from {memory_path}")
            with open(memory_path, "r") as f:
                new_memory = [self._json_to_memory(memory) for memory in json.load(f)]
            self._add_to_memory(new_memory)
            self.save_memory()
            return 

        if len(solutions) == 1:
            self._extract_from_single_solution(task, solutions[0], epoch)
        else:
            self._extract_from_single_task_multiple_solutions(task, solutions, epoch)
    
    def _process_response_into_memory_items(self, response: str) -> List[MemoryItem]:
        # Remove fenced code blocks markers if present and normalize lines
        lines = []
        for raw_line in response.splitlines():
            line = raw_line.strip("\n\r")
            if line.strip().startswith("```"):
                # skip markdown fence lines entirely
                continue
            lines.append(line)

        items: List[MemoryItem] = []

        current_title: str = ""
        current_description: str = ""
        current_content_lines: List[str] = []
        in_content: bool = False

        def maybe_flush_item():
            nonlocal current_title, current_description, current_content_lines, in_content
            if current_title or current_description or current_content_lines:
                title = current_title.strip()
                description = current_description.strip()
                content = "\n".join([l.rstrip() for l in current_content_lines]).strip()
                if title and description and content:
                    items.append(MemoryItem(title=title, description=description, content=content))
            current_title = ""
            current_description = ""
            current_content_lines = []
            in_content = False

        for line in lines:
            stripped = line.strip()
            # Start of a new memory item
            if stripped.startswith("# ") and "Memory Item" in stripped:
                maybe_flush_item()
                continue

            if stripped.startswith("## Title"):
                in_content = False
                # Capture text after '## Title'
                parts = stripped.split("## Title", 1)
                current_title = parts[1].strip() if len(parts) > 1 else ""
                continue

            if stripped.startswith("## Description"):
                in_content = False
                parts = stripped.split("## Description", 1)
                current_description = parts[1].strip() if len(parts) > 1 else ""
                continue

            if stripped.startswith("## Content"):
                # Enter content capture mode; everything until next header/new item belongs to content
                in_content = True
                parts = stripped.split("## Content", 1)
                first_content = parts[1].strip() if len(parts) > 1 else ""
                if first_content:
                    current_content_lines.append(first_content)
                continue

            if in_content:
                # Continue aggregating content; stop only when a new header or item begins
                if stripped.startswith("## ") or (stripped.startswith("# ") and "Memory Item" in stripped):
                    # This would be handled on next iteration, but to be safe, flush then re-handle
                    maybe_flush_item()
                    # Re-process this line in the outer loop by treating it as a header on next iteration
                    # Achieve by setting placeholders then continuing
                    # However, since we cannot re-inject the line, rely on next header detection to reset state
                    # and accept possible minor content loss for malformed inputs
                    continue
                current_content_lines.append(line)

        # Flush last item if present
        maybe_flush_item()

        return items

    def _extract_from_single_solution(self, task: Task, solution: str, epoch: int) -> None:
        system_prompt = f"""
You are an expert in GPU kernel programming. You will be given a task description and a solution attempt.
The solution attempt may be correct or incorrect, and may be faster or slower than other solutions.
## Guidelines
Your goal is to extract and summarize the most useful insights in the format of memory items.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.
## Important notes
- You must think why the solution is correct or incorrect, and why it is faster or slower than other solutions.
- You can extract at most 3 memory items.
- Do not repeat similar or overlapping items.
- Do not mention specific code snippets or task descriptions — focus on generalizable behaviors and reasoning patterns.
## Output Format
Your output must strictly follow the Markdown format shown below:
```# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task> ```
        """

        task_description = task.task_description

        prompt = f"""
Problem:
{task_description}

Solution:
{solution}
"""
        
        full_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        try:
            response = self.llm_client.chat_completion(full_prompt)["choices"][0]["message"]["content"]

            entries = self._process_response_into_memory_items(response)
            memory_path = os.path.join(self.run_dir, f"{task.task_id}_epoch_{epoch}_memory.json")
            with open(memory_path, "w") as f:
                json.dump([self._memory_to_json(entry) for entry in entries], f, indent=2)
            self._add_to_memory(entries)
            self.save_memory()
        except Exception as e:
            logger.error(f"Error extracting memory: {e}")


    def _extract_from_single_task_multiple_solutions(self, task: Task, solutions: List[str], epoch: int) -> None:
        system_prompt = f"""
You are an expert in GPU kernel programming. You will be given a task description and multiple solution attempts. 
Some solutions may be correct, and others may be incorrect. Some solutions may be faster than others.
## Guidelines
Your goal is to compare and contrast these solutions to identify the most useful and generalizable strategies as memory items.
Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed solutions and formulate preventative strategies.
- Prefer strategies that generalize beyond specific tasks.
## Important notes
- Think first: Why did some solutions succeed while others failed? Why are some solutions faster than others?
- You can extract at most 5 memory items total.
- Do not repeat similar or overlapping items.
- Do not mention specific code snippets or task descriptions — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.
## Output Format
Your output must strictly follow the Markdown format shown below:
```# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task> ```
        """

        task_description = task.task_description

        prompt = f"""
Problem:
{task_description}
"""
        for i, trajectory in enumerate(solutions):
            prompt += f"\nSolution {i+1}:\n{trajectory}"
        
        full_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        response = self.llm_client.chat_completion(full_prompt)["choices"][0]["message"]["content"]

        entries = self._process_response_into_memory_items(response)

        memory_path = os.path.join(self.run_dir, f"{task.task_id}_epoch_{epoch}_memory.json")
        with open(memory_path, "w") as f:
            json.dump([self._memory_to_json(entry) for entry in entries], f, indent=2)

        self._add_to_memory(entries)
        self.save_memory()



# class Rules(KnowledgeBase):
#     def __init__(self):
#         self.rules = []

#     def retrieve(self, query):
#         return "\n".join(self.rules)

#     def add(self, entry):
#         self.rules.append(entry)
    
#     def extract(self, solutions: List[Solution], evals: List[EvaluationResult]) -> List[str]:
#         pass

def get_memory(config, run_dir: str):
    if config.memory == "memory":
        return Memory(config, run_dir)
    else:
        raise ValueError(f"Invalid memory type: {config.memory}")