"""
Prompt dataset for the GRPO "write good Python" environment.

Each task is a natural-language instruction asking the model to write a small,
self-contained Python function or class. Tasks are intentionally simple so that
a well-behaved policy can produce code that compiles cleanly, passes pyflakes /
pycodestyle, and contains no obvious security issues.

The instructions deliberately bias the model toward clean, secure code (PEP 8,
no ``eval``/``exec``/``shell=True``) since those are exactly what the reward
function measures.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are an expert Python engineer. Write clean, secure, PEP 8 compliant "
    "Python code. Avoid insecure constructs such as eval, exec, pickle on "
    "untrusted data, or subprocess with shell=True. Respond with a single "
    "Python code block and nothing else."
)

TASKS: list[str] = [
    "Write a function `is_palindrome(text)` that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters.",
    "Write a function `fibonacci(n)` that returns a list of the first n Fibonacci numbers.",
    "Write a function `flatten(nested)` that flattens an arbitrarily nested list of integers into a single flat list.",
    "Write a function `word_count(text)` that returns a dictionary mapping each word to how many times it appears, case-insensitively.",
    "Write a function `merge_sorted(a, b)` that merges two already-sorted lists into one sorted list without using sorted().",
    "Write a function `read_json_file(path)` that safely loads and returns the JSON content of a file, raising a clear error if the file is missing.",
    "Write a class `Stack` with push, pop, peek, and is_empty methods, raising IndexError on pop/peek from an empty stack.",
    "Write a function `chunk(items, size)` that splits a list into consecutive chunks of the given size.",
    "Write a function `safe_divide(a, b)` that returns a / b, or None if b is zero.",
    "Write a function `count_vowels(text)` that returns the number of vowels in a string.",
    "Write a function `unique_preserve_order(items)` that removes duplicates from a list while preserving the original order.",
    "Write a function `roman_to_int(roman)` that converts a Roman numeral string to its integer value.",
    "Write a function `group_by_parity(numbers)` that returns a dict with keys 'even' and 'odd' mapping to lists of numbers.",
    "Write a class `Counter` that tracks integer counts per key, with increment(key) and most_common(n) methods.",
    "Write a function `transpose(matrix)` that returns the transpose of a 2D list (a list of rows).",
    "Write a function `validate_email(address)` that returns True for a basic, well-formed email address using the re module.",
    "Write a function `running_average(numbers)` that yields the running average after each element using a generator.",
    "Write a function `hash_password(password, salt)` that securely hashes a password using hashlib with the provided salt.",
    "Write a function `temperature_convert(value, to_unit)` that converts between Celsius and Fahrenheit based on to_unit ('C' or 'F').",
    "Write a function `binary_search(sorted_list, target)` that returns the index of target or -1 if absent.",
]


def build_prompt(tokenizer, task: str) -> str:
    """Render a task into a model-ready prompt string.

    Uses the tokenizer's chat template when available (instruct models),
    otherwise falls back to a plain instruction format.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n# Task:\n# {task}\n\n```python\n"
