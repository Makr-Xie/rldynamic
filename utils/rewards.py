#!/usr/bin/env python3
"""
Custom reward function
"""

import re


_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str):
    last_boxed_str = last_boxed_only_string(solution_str)

    if last_boxed_str is None:
        return None

    answer = remove_boxed(last_boxed_str)
    cleaned_answer = strip_string(answer)
    return cleaned_answer


def compute_correctness_score(solution_str, ground_truth, **kwargs):
    answer = extract_solution(solution_str=solution_str)

    if answer is not None and answer == ground_truth:
        return 1.0
    return 0.0


def compute_format_score(solution_str, **kwargs):
    score = 0
    lower_text = solution_str.lower()
    no_space_text = lower_text.replace(" ", "")
    
    if re.search(r'\bfirst\b', lower_text):
        score += 0.25

    if re.search(r'\bsecond\b', lower_text):
        score += 0.25
        
    if re.search(r'\bthird\b', lower_text):
        score += 0.25

    if 'step1' in no_space_text:
        score += 0.25
        
    if 'step2' in no_space_text:
        score += 0.25
        
    if r'\boxed' in lower_text:
        score += 0.4

    return min(score, 1.0)


def compute_length_score(solution_str, max_length=800, **kwargs):
    response_len = len(solution_str)
    score = 1 - abs(response_len / max_length - 1)
    return max(score, 0.0)


# Default weights for combining scores
DEFAULT_WEIGHTS = {
    'correctness': 0.34,
    'format': 0.33,
    'length': 0.33
}


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  weights=None, **kwargs):
    """
    Compute the combined reward score using all three components with weights.
    This is the main function used for final model updates.

    Args:
        data_source: Data source identifier
        solution_str: The model's generated solution
        ground_truth: The correct answer
        extra_info: Optional extra information
        weights: Optional dict with weights for each component.
                If None, uses DEFAULT_WEIGHTS

    Returns:
        float: Total weighted score
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Compute individual binary components
    correctness = compute_correctness_score(solution_str, ground_truth, **kwargs)
    format_score = compute_format_score(solution_str, **kwargs)
    length_score = compute_length_score(solution_str, **kwargs)

    # Apply weights and combine
    total_score = (
        correctness * weights['correctness'] +
        format_score * weights['format'] +
        length_score * weights['length']
    )

    return total_score


def compute_component_scores(data_source, solution_str, ground_truth,
                            extra_info=None, weights=None, **kwargs):
    """
    Compute all three reward components separately and return them as a dictionary.
    This function is useful for gradient analysis where we need individual components.

    Args:
        data_source: Data source identifier
        solution_str: The model's generated solution
        ground_truth: The correct answer
        extra_info: Optional extra information
        weights: Optional dict with weights for each component

    Returns:
        dict: Dictionary containing binary scores, weighted scores, and total
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Compute binary scores
    correctness = compute_correctness_score(solution_str, ground_truth, **kwargs)
    format_score = compute_format_score(solution_str, **kwargs)
    length_score = compute_length_score(solution_str, **kwargs)

    # Calculate weighted scores
    weighted_correctness = correctness * weights['correctness']
    weighted_format = format_score * weights['format']
    weighted_length = length_score * weights['length']

    return {
        # Binary scores (0 or 1)
        'correctness_binary': correctness,
        'format_binary': format_score,
        'length_binary': length_score,

        # Weighted scores
        'correctness_weighted': weighted_correctness,
        'format_weighted': weighted_format,
        'length_weighted': weighted_length,

        # Weights used
        'weights': weights,

        # Total score
        'total': weighted_correctness + weighted_format + weighted_length
    }




# ============================================================================
# helper fuctions for math reward computation
# ============================================================================

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    if s.startswith(left):
        return s[len(left):-1]
        
    left = "\\fbox{"
    if s.startswith(left):
        return s[len(left):-1]

    return s


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    if string[idx:idx+7] == "\\boxed ":
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    
    return string[idx : right_brace_idx + 1]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
    except ValueError:
        pass
    return string


def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    string = re.sub(r"\\sqrt(?![{\[])([a-zA-Z0-9])", r"\\sqrt{\1}", string)
    return string

def strip_string(string: str) -> str:
    string = string.replace("\n", "").replace("\\!", "")
    string = string.replace("\\\\", "\\").replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("\\%", "").replace("\%", "")
    string = remove_right_units(string)
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) > 0 and string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string