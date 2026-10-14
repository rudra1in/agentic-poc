"""
problems/bank.py — DSA problem bank

Merged from AURA++ ProblemTool + lumina practice_agent.
Problems are stored in data/problems.json (auto-created with defaults).
`get_problem()` filters by topic/difficulty and avoids already-seen IDs.
"""

import json
import random
from pathlib import Path
from typing import Optional

BASE_DIR      = Path(__file__).resolve().parents[1]
PROBLEMS_FILE = BASE_DIR / "data" / "problems.json"

# ----------------------------------------------------------------
# Default built-in problems (seed data)
# ----------------------------------------------------------------
DEFAULT_PROBLEMS: list[dict] = [
    # Arrays
    {"id": "arr_001", "title": "Two Sum", "topic": "arrays", "difficulty": "easy",
     "description": "Given an array of integers `nums` and an integer `target`, return the indices of the two numbers that add up to `target`. Each input has exactly one solution."},
    {"id": "arr_002", "title": "Best Time to Buy and Sell Stock", "topic": "arrays", "difficulty": "easy",
     "description": "Given an array `prices` where `prices[i]` is the price of a stock on day `i`, find the maximum profit you can achieve by choosing a single day to buy and a single later day to sell."},
    {"id": "arr_003", "title": "Container With Most Water", "topic": "arrays", "difficulty": "medium",
     "description": "Given `n` non-negative integers representing heights of vertical lines, find two lines that together with the x-axis form a container that holds the most water."},
    # Strings
    {"id": "str_001", "title": "Longest Substring Without Repeating Characters", "topic": "strings", "difficulty": "medium",
     "description": "Given a string `s`, find the length of the longest substring without repeating characters."},
    {"id": "str_002", "title": "Valid Anagram", "topic": "strings", "difficulty": "easy",
     "description": "Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`, and `false` otherwise."},
    # Dynamic Programming
    {"id": "dp_001", "title": "House Robber", "topic": "dynamic_programming", "difficulty": "medium",
     "description": "You are a robber planning to rob houses along a street. Adjacent houses have security systems. Given an array of non-negative integers `nums`, return the maximum amount you can rob without alerting the police."},
    {"id": "dp_002", "title": "Climbing Stairs", "topic": "dynamic_programming", "difficulty": "easy",
     "description": "You are climbing a staircase with `n` steps. Each time you can climb 1 or 2 steps. In how many distinct ways can you climb to the top?"},
    {"id": "dp_003", "title": "Coin Change", "topic": "dynamic_programming", "difficulty": "medium",
     "description": "Given an array of coin denominations and a total `amount`, return the fewest number of coins needed to make up that amount. Return -1 if it cannot be done."},
    {"id": "dp_004", "title": "Longest Common Subsequence", "topic": "dynamic_programming", "difficulty": "medium",
     "description": "Given two strings `text1` and `text2`, return the length of their longest common subsequence. A subsequence is a sequence derived by deleting some characters without changing the order of the remaining characters."},
    # Trees
    {"id": "tree_001", "title": "Maximum Depth of Binary Tree", "topic": "trees", "difficulty": "easy",
     "description": "Given the root of a binary tree, return its maximum depth (the number of nodes along the longest path from the root to a leaf node)."},
    {"id": "tree_002", "title": "Validate Binary Search Tree", "topic": "trees", "difficulty": "medium",
     "description": "Given the root of a binary tree, determine if it is a valid binary search tree (BST)."},
    {"id": "tree_003", "title": "Binary Tree Level Order Traversal", "topic": "trees", "difficulty": "medium",
     "description": "Given the root of a binary tree, return the level order traversal of its nodes' values (i.e., from left to right, level by level)."},
    # Graphs
    {"id": "graph_001", "title": "Number of Islands", "topic": "graphs", "difficulty": "medium",
     "description": "Given an m×n grid of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and formed by connecting adjacent land cells horizontally or vertically."},
    {"id": "graph_002", "title": "Course Schedule", "topic": "graphs", "difficulty": "medium",
     "description": "There are `numCourses` courses labeled 0 to numCourses-1. Given a list of prerequisite pairs, determine if you can finish all courses (i.e., detect if the prerequisite graph has a cycle)."},
    # Binary Search
    {"id": "bs_001", "title": "Binary Search", "topic": "binary_search", "difficulty": "easy",
     "description": "Given a sorted array of integers `nums` and an integer `target`, return the index of `target` or -1 if not found. You must write an O(log n) algorithm."},
    {"id": "bs_002", "title": "Find Minimum in Rotated Sorted Array", "topic": "binary_search", "difficulty": "medium",
     "description": "Given a sorted array that has been rotated between 1 and n times, find the minimum element. You must write an O(log n) algorithm."},
    # Linked List
    {"id": "ll_001", "title": "Reverse Linked List", "topic": "linked_list", "difficulty": "easy",
     "description": "Given the head of a singly linked list, reverse the list and return the reversed list."},
    {"id": "ll_002", "title": "Detect Cycle in Linked List", "topic": "linked_list", "difficulty": "easy",
     "description": "Given the head of a linked list, determine if the linked list has a cycle using O(1) extra memory."},
    # Greedy
    {"id": "greedy_001", "title": "Jump Game", "topic": "greedy", "difficulty": "medium",
     "description": "You are given an integer array `nums` where `nums[i]` is the maximum jump length from index `i`. Return `true` if you can reach the last index starting from index 0."},
    # Backtracking
    {"id": "bt_001", "title": "Subsets", "topic": "backtracking", "difficulty": "medium",
     "description": "Given an integer array `nums` of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets."},
    {"id": "bt_002", "title": "N-Queens", "topic": "backtracking", "difficulty": "hard",
     "description": "Place `n` queens on an n×n chessboard such that no two queens attack each other. Return all distinct solutions."},
]


def _load_problems() -> list[dict]:
    if PROBLEMS_FILE.exists():
        try:
            return json.loads(PROBLEMS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Seed file on first run
    PROBLEMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROBLEMS_FILE.write_text(json.dumps(DEFAULT_PROBLEMS, indent=2), encoding="utf-8")
    return DEFAULT_PROBLEMS


def get_problem(
    topic: Optional[str] = None,
    difficulty: Optional[str] = None,
    exclude_ids: Optional[set[str]] = None,
) -> Optional[dict]:
    """
    Return a random problem matching the filters.
    Falls back to relaxing difficulty, then topic, before giving up.
    """
    problems     = _load_problems()
    exclude_ids  = exclude_ids or set()

    def _filter(t: Optional[str], d: Optional[str]) -> list[dict]:
        pool = [p for p in problems if p["id"] not in exclude_ids]
        if t:
            pool = [p for p in pool if p.get("topic") == t]
        if d:
            pool = [p for p in pool if p.get("difficulty") == d]
        return pool

    # Strict match first
    pool = _filter(topic, difficulty)
    if pool:
        return random.choice(pool)

    # Relax difficulty
    pool = _filter(topic, None)
    if pool:
        return random.choice(pool)

    # Relax topic
    pool = _filter(None, difficulty)
    if pool:
        return random.choice(pool)

    # Any problem not seen
    pool = [p for p in problems if p["id"] not in exclude_ids]
    return random.choice(pool) if pool else None


def list_topics() -> list[str]:
    return sorted({p["topic"] for p in _load_problems()})
