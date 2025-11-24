# Project

### Plan
- [ ] Baseline test-time scaling methods (best-of-N, iterative refinement)
    - [ ] Qwen2.5-Coder-7B-Instruct 
        - [ ] Level 1
        - [ ] Level 2
    - [ ] QwQ-32B
        - [ ] Level 1
        - [ ] Level 2
- [ ] AutoRule rule extraction with QwQ-32B
- [ ] Use rules from prev step as prompt
	- [ ] Use rules as part of feedback for iterative refinement (new)
- [ ] Collect good kernels from prev steps and run SFT on 7B model (rejection sampling)
- [ ] RL training with correctness/speedup as reward (Kevin) - stretch
- [ ] RL training with + rules as reward (AutoRule) - stretch