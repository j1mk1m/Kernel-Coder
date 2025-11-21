from dataclasses import dataclass
from typing import List, Tuple
import logging
import os
import json
import numpy as np
import hashlib
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from KernelCoder.benchmarks.benchmark import Task, Solution, EvaluationResult
from llm_utils import create_llm_client


class KnowledgeBase:
    def __init__(self):
        pass

    def retrieve(self, query):
        pass

    def extract(self, trajectories: dict[Task, List[Tuple[Task, Solution, EvaluationResult, str]]], **kwargs) -> None:
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
        self.query_embeddings_path = os.path.join(self.run_dir, "query_embeddings.json")
        self.query_embeddings = {}
        self.load_memory()
        self.config = config
        base_api_url = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
        self.llm_client = create_llm_client(data_file=os.path.join(self.run_dir, "memory_llm_usage.json"), default_model=config.memory_model_name, default_temperature=1.0, default_api_base=base_api_url, default_max_tokens=config.max_tokens)
        self.embedding_model = create_llm_client(data_file=os.path.join(self.run_dir, "memory_embedding_usage.json"), default_model=config.memory_embedding_model_name)
    
    def _add_to_memory(self, memory_items: List[MemoryItem]) -> None:
        for memory_item in memory_items:
            self.memory.append(memory_item)
            self.embeddings.append(self._embed(memory_item))
    
    def save_memory(self) -> None:
        with open(self.memory_path, "w") as f:
            json.dump([self._memory_to_json(memory) for memory in self.memory], f, indent=2)
        with open(self.embeddings_path, "w") as f:
            json.dump(self.embeddings, f, indent=2)
        with open(self.query_embeddings_path, "w") as f:
            json.dump(self.query_embeddings, f, indent=2)
        self.llm_client.save_usage_data()
        self.embedding_model.save_usage_data()
    
    def load_memory(self) -> None:
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r") as f:
                self.memory = [self._json_to_memory(memory) for memory in json.load(f)]
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, "r") as f:
                self.embeddings = json.load(f)
        if os.path.exists(self.query_embeddings_path):
            with open(self.query_embeddings_path, "r") as f:
                self.query_embeddings = json.load(f)
    
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
        # compute hash of query and check if it exists in query_embeddings
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.query_embeddings:
            query_embedding = self.query_embeddings[query_hash]
        else:
            query_embedding = self.embedding_model.embedding(query)["data"][0]["embedding"]
            self.query_embeddings[query_hash] = query_embedding
        similarities = [np.dot(query_embedding, embedding) for embedding in self.embeddings]
        relevant_memory = [self.memory[np.argsort(similarities)[-1]]] # TODO: change to top k, top 1 for now
        answer = "Consider the following tips:" + "\n".join([self._memory_to_string(memory) for memory in relevant_memory])
        return answer

    def extract(self, epoch: int, trajectories: dict[str, List[Tuple[Task, Solution, EvaluationResult, str]]], **kwargs) -> None:
        logger.info(f"Extracting memory")
        assert len(trajectories) == 1, "Memory extraction only supports a single task at a time"
        task_id, trajectories = list(trajectories.items())[0]
        task = trajectories[0][0]
        trajectories = [trajectory for _, solution, evaluation, trajectory in trajectories]
        memory_path = os.path.join(self.run_dir, f"{task.task_id}_epoch_{epoch}_memory.json")
        if os.path.exists(memory_path):
            print(f"Memory already exists for {task.task_id} at epoch {epoch}")
            # with open(memory_path, "r") as f:
            #     new_memory = [self._json_to_memory(memory) for memory in json.load(f)]
            # self._add_to_memory(new_memory)
            # self.save_memory()
            return 

        if len(trajectories) == 1:
            self._extract_from_single_solution(task, trajectories[0], epoch)
        else:
            self._extract_from_single_task_multiple_solutions(task, trajectories, epoch)
    
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
            response = self.llm_client.chat_completion(full_prompt, tag="memory_extraction")["choices"][0]["message"]["content"]

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

        response = self.llm_client.chat_completion(full_prompt, tag="memory_extraction")["choices"][0]["message"]["content"]

        entries = self._process_response_into_memory_items(response)

        memory_path = os.path.join(self.run_dir, f"{task.task_id}_epoch_{epoch}_memory.json")
        with open(memory_path, "w") as f:
            json.dump([self._memory_to_json(entry) for entry in entries], f, indent=2)

        self._add_to_memory(entries)
        self.save_memory()


class Rules(KnowledgeBase):
    def __init__(self, config, run_dir: str):
        self.rules = []
        self.config = config
        self.run_dir = run_dir
        base_api_url = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
        self.llm_client = create_llm_client(data_file=os.path.join(self.run_dir, "rules_llm_usage.json"), default_model=config.memory_model_name, default_temperature=1.0, default_api_base=base_api_url, default_max_tokens=config.max_tokens)

    def retrieve(self, query):
        return "When writing kernels, consider the following tips:\n" + "\n".join(self.rules)

    def _rule_is_satisfied(self, rule, kernel_src):
        prompt = f"""You are a kernel expert. Determine whether the following kernel satisfies the following rule.
    {rule}

    Be as objective as possible when evaluating the rule and do not evaluate other characteristics of the response. If the rule is not applicable for this task, treat it as if the rule is satisfied. 
    You must provide your answer by strictly outputting either one of the following two options:"[[Yes]]" or "[[No]]" and nothing else

    Kernel:
    {kernel_src}
    """
        response = self.llm_client.text_completion(prompt, tag="rule_alignment")
        response = response["choices"][0]["text"]
        return "Yes" in response


    def extract(self, epoch: int, trajectories: dict[str, List[Tuple[Task, Solution, EvaluationResult, str]]], batch_num=0, **kwargs) -> None:
        batch_dir = os.path.join(self.run_dir, f"epoch_{epoch}_batch_{batch_num}")
        os.makedirs(batch_dir, exist_ok=True)
        all_rules = []
        logger.info(f"Extracting rules for epoch {epoch} batch {batch_num}")
        for task_id, trajectory in trajectories.items():
            task = trajectory[0][0]
            traj_string = "\n".join([t for task, s, e, t in trajectory])
            file_path = os.path.join(batch_dir, f"{task_id}_comparative_analysis.txt")
            if os.path.exists(file_path):
                response = open(file_path, "r").read()
            else:
                prompt = f"""You are a kernel expert. You are given a task description and multiple solutions. Some solutions may be correct, and others may be incorrect. Some solutions may be faster than others. Analyze why some solutions are correct and others are incorrect, and why some solutions are faster than others.
Task description:
{task.task_description}

Solutions:
{traj_string}
    """
                response = self.llm_client.text_completion(prompt, tag="rule_extraction")["choices"][0]["text"]

                with open(file_path, "w") as f:
                    f.write(response)

            file_path = os.path.join(batch_dir, f"{task.task_id}_rules.txt")
            if os.path.exists(file_path):
                response = open(file_path, "r").read()
            else:
                prompt = f"""Based on the following comparative analysis, extract any rule-like statements implied by the analysis to indicate the difference. Rule-like statements should be ablet to be judged objectively and determinsitcially. The rules shoud be general enough to be applied to various CUDA kernels. Below are few examples of rule-like statements:
Example 1:
- The kernel performs operator fusion between multiple operations.
Example 2:
- The kernel uses shared memory tiling to reduce global memory access.
Example 3:
- The kernel uses thread block sizes that are multiples of warp size (32).
Return the list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. If there are no rule-like statements, return an empty JSON array. List at most 3 rules.

[Reasoning]
{response}
"""

                response = self.llm_client.text_completion(prompt, tag="rule_extraction")["choices"][0]["text"]
                with open(file_path, "w") as f:
                    f.write(response)

            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()

                rules = json.loads(response)
            except Exception as e:
                logger.error(f"Error parsing rule response for {task.task_id}: {e}")
                rules = []
            
            with open(os.path.join(batch_dir, f"{task.task_id}_rules.json"), "w") as f:
                json.dump(rules, f, indent=2)
            
            all_rules.extend(rules)

        logger.info("Merging rules")
        file_path = os.path.join(batch_dir, "merged_rules.txt")
        if os.path.exists(file_path):
            rule_response = open(file_path, "r").read()
        else:
            rules_str = "\n".join(all_rules)
            prompt = f"""Below is a large list of rule-like statements regarding the behavior of CUDA kernels. Some of these rules might be duplicates or very similar.
Please merge them so that there are no duplicates or very similar rules. Condense the rules into at most 25 rules.
Return the merged list as a JSON array of strings. Do not use ``json``, just output the JSON array directly. 
[Rules]
{rules_str}
"""

            with open(os.path.join(batch_dir, "rule_merging_prompt.txt"), "w") as f:
                f.write(prompt)
            rule_response = self.llm_client.text_completion(prompt, max_tokens=32768, tag="rule_merging")
            rule_response = rule_response["choices"][0]["text"]
            with open(file_path, "w") as f:
                f.write(rule_response)

        try:
            if "```json" in rule_response:
                rule_response = rule_response.split("```json")[1].split("```")[0].strip()

            rules = json.loads(rule_response)

        except Exception as e:
            logger.error(f"Error parsing merged rule response: {e}")
            rules = []

        with open(os.path.join(batch_dir, "merged_rules.json"), "w") as f:
            json.dump(rules, f, indent=2)

        logger.info("Aligning rules")
        alignment_dir = os.path.join(batch_dir, "alignment")
        os.makedirs(alignment_dir, exist_ok=True)
        rule_alignment_results = []
        for rule_num, rule in enumerate(rules):
            aligned = 0
            not_aligned = 0
            both_false = 0
            both_true = 0
            total = 0
            if os.path.exists(os.path.join(alignment_dir, f"rule_{rule_num}_alignment.json")):
                result = json.load(open(os.path.join(alignment_dir, f"rule_{rule_num}_alignment.json")))
            else:
                tmp_trajectories = list(trajectories.items())
                if len(tmp_trajectories) > self.config.autorule_num_alignment_samples:
                    tmp_trajectories = random.sample(tmp_trajectories, self.config.autorule_num_alignment_samples)
                for task, trajectory in tmp_trajectories:
                    rule_satisfied = []
                    for (_, s, e, t) in trajectory:
                        satisfied = self._rule_is_satisfied(rule, s.solution_code)
                        rule_satisfied.append((s, e, t, satisfied))
                    
                    # Compare every pair of solutions
                    for i in range(len(rule_satisfied)):
                        for j in range(i + 1, len(rule_satisfied)):
                            (s1, e1, t1, satisfied1) = rule_satisfied[i]
                            (s2, e2, t2, satisfied2) = rule_satisfied[j]
                            if satisfied1 and satisfied2:
                                both_true += 1
                            elif not satisfied1 and not satisfied2:
                                both_false += 1
                            elif satisfied1 and not satisfied2:
                                if e1 > e2:
                                    aligned += 1
                                else:
                                    not_aligned += 1
                            elif not satisfied1 and satisfied2:
                                if e2 > e1:
                                    aligned += 1
                                else:
                                    not_aligned += 1
                            total += 1

                alignment_rate = aligned / (aligned + not_aligned) if aligned + not_aligned > 0 else 0.0
                result = {"rule": rule, "total": total, "aligned": aligned, "alignment_rate": alignment_rate, "both_false": both_false, "both_true": both_true}
                with open(os.path.join(alignment_dir, f"rule_{rule_num}_alignment.json"), "w") as f:
                    json.dump(result, f, indent=2)
            rule_alignment_results.append(result)
        
        # Filter rules based on alignment rate
        filtered_rules = [result["rule"] for result in rule_alignment_results if result["alignment_rate"] >= self.config.autorule_alignment_threshold]

        with open(os.path.join(batch_dir, "filtered_rules.json"), "w") as f:
            json.dump(filtered_rules, f, indent=2)

        self.rules.extend(filtered_rules)
        self.llm_client.save_usage_data()


def get_memory(config, run_dir: str):
    if config.memory == "memory":
        return Memory(config, run_dir)
    elif config.memory == "rules":
        return Rules(config, run_dir)
    else:
        raise ValueError(f"Invalid memory type: {config.memory}")