# Project

### Plan
- [x] Baseline test-time scaling methods (best-of-N, iterative refinement on Qwen2.5-Coder-7B-Instruct and QwQ-32B)
- [ ] AutoRule rule extraction with QwQ-32B
- [ ] Use rules from prev step as prompt
	- [ ] Use rules as part of feedback for iterative refinement (new)
- [ ] Collect good kernels from prev steps and run SFT on 7B model (rejection sampling)
- [ ] RL training with correctness/speedup as reward (Kevin) - stretch
- [ ] RL training with + rules as reward (AutoRule) - stretch