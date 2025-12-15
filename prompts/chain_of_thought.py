"""
Chain of Thought (CoT) Prompting Examples
Testing if AIBOM can detect reasoning and CoT patterns.
"""

# =============================================================================
# BASIC CHAIN OF THOUGHT PROMPTS
# =============================================================================

BASIC_COT_PROMPT = """Let's think through this step by step.

Question: {question}

Step-by-step reasoning:"""

ZERO_SHOT_COT_PROMPT = """Q: {question}

A: Let's approach this step by step:
1."""

# =============================================================================
# MATHEMATICAL REASONING COT
# =============================================================================

MATH_COT_PROMPT = """Solve this math problem step by step. Show your work.

Problem: {problem}

Solution:
Let me break this down:
Step 1:"""

MATH_COT_EXAMPLE = """Solve this math problem step by step. Show your work.

Problem: If a train travels at 60 mph for 2.5 hours, how far does it travel?

Solution:
Let me break this down:
Step 1: Identify what we know
- Speed = 60 miles per hour
- Time = 2.5 hours

Step 2: Recall the formula
- Distance = Speed × Time

Step 3: Calculate
- Distance = 60 mph × 2.5 hours
- Distance = 150 miles

Therefore, the train travels 150 miles.

---

Problem: {problem}

Solution:
Let me break this down:
Step 1:"""

# =============================================================================
# LOGICAL REASONING COT
# =============================================================================

LOGICAL_REASONING_COT = """Let's analyze this logical problem carefully.

Problem: {problem}

Analysis:
First, let me identify the key elements:
-"""

LOGICAL_COT_EXAMPLE = """Let's analyze this logical problem carefully.

Problem: All cats are mammals. Some mammals are pets. Can we conclude that all cats are pets?

Analysis:
First, let me identify the key elements:
- Premise 1: All cats are mammals (cats ⊆ mammals)
- Premise 2: Some mammals are pets (mammals ∩ pets ≠ ∅)
- Conclusion to evaluate: All cats are pets (cats ⊆ pets)?

Let me think through this:
1. From Premise 1: Every cat is definitely a mammal
2. From Premise 2: At least one mammal is a pet, but not all mammals need to be pets
3. The premises don't tell us anything specific about cats and pets

The logical fallacy here is:
- Just because cats are mammals, and some mammals are pets
- Doesn't mean that cats specifically are in the "pet" subset of mammals

Conclusion: No, we CANNOT conclude that all cats are pets. This is a logical fallacy known as "undistributed middle."

---

Problem: {problem}

Analysis:
First, let me identify the key elements:
-"""

# =============================================================================
# CODE DEBUGGING COT
# =============================================================================

CODE_DEBUGGING_COT = """Let's debug this code step by step.

Code:
```{language}
{code}
```

Error/Issue: {error}

Debugging process:
1. First, I'll understand what the code is trying to do:
"""

CODE_DEBUG_EXAMPLE = """Let's debug this code step by step.

Code:
```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([])
```

Error/Issue: ZeroDivisionError: division by zero

Debugging process:
1. First, I'll understand what the code is trying to do:
   - It's calculating the average of a list of numbers
   - Sums all numbers, then divides by count

2. Let me trace through with the given input []:
   - numbers = [] (empty list)
   - total = 0 (loop doesn't execute)
   - len(numbers) = 0
   - total / len(numbers) = 0 / 0 = ERROR!

3. Root cause identified:
   - The function doesn't handle empty lists
   - Division by zero when list is empty

4. Solution:
```python
def calculate_average(numbers):
    if not numbers:  # Check for empty list
        return 0  # or raise ValueError("Cannot calculate average of empty list")
    total = sum(numbers)
    return total / len(numbers)
```

---

Code:
```{language}
{code}
```

Error/Issue: {error}

Debugging process:
1. First, I'll understand what the code is trying to do:
"""

# =============================================================================
# SELF-CONSISTENCY COT (Multiple reasoning paths)
# =============================================================================

SELF_CONSISTENCY_COT = """I'll solve this problem using multiple approaches to verify my answer.

Problem: {problem}

Approach 1:
{approach_placeholder}

Approach 2 (different method):
{approach_placeholder}

Approach 3 (verification):
{approach_placeholder}

Comparing results:
- Approach 1 gives:
- Approach 2 gives:
- Approach 3 gives:

Final answer (most consistent):"""

# =============================================================================
# TREE OF THOUGHT (ToT) PROMPTS
# =============================================================================

TREE_OF_THOUGHT_PROMPT = """Explore multiple solution paths for this problem.

Problem: {problem}

Let me consider different approaches:

Branch A - [First approach]:
  Pros:
  Cons:
  Evaluation score: /10

Branch B - [Second approach]:
  Pros:
  Cons:
  Evaluation score: /10

Branch C - [Third approach]:
  Pros:
  Cons:
  Evaluation score: /10

Best path forward: Branch _
Reasoning:"""

# =============================================================================
# REFLECTION AND SELF-CRITIQUE COT
# =============================================================================

REFLECTION_COT_PROMPT = """Solve the problem, then reflect on your solution.

Problem: {problem}

Initial Solution:
[Provide solution]

Reflection:
1. What assumptions did I make?
2. Are there edge cases I missed?
3. Is my reasoning sound?
4. Could this be wrong? How?

Refined Solution (if needed):
[Provide refined solution based on reflection]

Final Answer:"""

SELF_CRITIQUE_COT = """Answer the question, then critique your own response.

Question: {question}

My Answer:
[Answer]

Self-Critique:
- Strengths of my answer:
- Weaknesses or gaps:
- What I might have missed:
- Confidence level (1-10):

Improved Answer (incorporating critique):"""

# =============================================================================
# ANALOGICAL REASONING COT
# =============================================================================

ANALOGICAL_REASONING_COT = """Let me solve this by finding a similar, simpler problem.

Problem: {problem}

Finding an analogy:
1. This problem is similar to: [simpler analogous problem]
2. In the simpler problem: [solution approach]
3. Mapping back to original: [apply same logic]

Solution using analogy:"""

# =============================================================================
# DECOMPOSITION COT
# =============================================================================

DECOMPOSITION_COT = """Break down this complex problem into smaller parts.

Problem: {problem}

Decomposition:
Sub-problem 1: [description]
  Solution:

Sub-problem 2: [description]
  Solution:

Sub-problem 3: [description]
  Solution:

Combining sub-solutions:

Final Answer:"""

# =============================================================================
# COT PROMPT REGISTRY
# =============================================================================

COT_PROMPTS = {
    "basic": BASIC_COT_PROMPT,
    "zero_shot": ZERO_SHOT_COT_PROMPT,
    "math": MATH_COT_PROMPT,
    "math_example": MATH_COT_EXAMPLE,
    "logical": LOGICAL_REASONING_COT,
    "logical_example": LOGICAL_COT_EXAMPLE,
    "code_debug": CODE_DEBUGGING_COT,
    "code_debug_example": CODE_DEBUG_EXAMPLE,
    "self_consistency": SELF_CONSISTENCY_COT,
    "tree_of_thought": TREE_OF_THOUGHT_PROMPT,
    "reflection": REFLECTION_COT_PROMPT,
    "self_critique": SELF_CRITIQUE_COT,
    "analogical": ANALOGICAL_REASONING_COT,
    "decomposition": DECOMPOSITION_COT,
}


def get_cot_prompt(prompt_type: str, **kwargs) -> str:
    """Get a chain-of-thought prompt by type."""
    template = COT_PROMPTS.get(prompt_type, BASIC_COT_PROMPT)
    if kwargs:
        return template.format(**kwargs)
    return template

