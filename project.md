# Project

### Plan
- [x] Baseline test-time scaling methods (best-of-N, iterative refinement)
- [ ] Stage 1: AutoRule rule extraction with QwQ-32B (Mon)
    - [ ] Level 1
    - [x] Level 2
- [ ] Stage 2: Rule-guided prompt optimization
	- [ ] Use rules as part of feedback for iterative refinement
- [ ] Stage 3: Rejection Fine-Tuning
    - Filter kernels by execution results and rule satisfaction
    - SFT on 7B model
    - compare performance to only filtering with execution results?