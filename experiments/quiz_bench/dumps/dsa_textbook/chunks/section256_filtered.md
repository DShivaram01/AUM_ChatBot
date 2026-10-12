# dsa_textbook — chunker `section256_filtered`

34 chunks; Mistral tokens min/median/max = 27/106/255; 4 chunks exceed the 500-char production prompt view.

## chunk 0 · p.1 · 27 tokens · Data Structures & Algorithms: Comprehensive Course Notes

```text
Course Code: CS-201 | Subject: Fundamentals of Data Structures, Algorithm Analysis, and Complexity Theory
```

## chunk 1 · p.1 · 99 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 1: Foundations of Algorithm Analysis & Asymptotic Notation

```text
An algorithm is a finite sequence of well-defined instructions designed to solve a specific computational problem. When evaluating algorithms, we focus on efficiency: Time Complexity (how runtime scales with input size n) and Space Complexity (how memory usage scales). Rather than measuring execution time in seconds—which depends heavily on hardware, compiler optimizations, and processor load—we analyze asymptotic growth using Big-O, BigOmega (Ω), and Big-Theta (Θ) notations.
```

## chunk 2 · p.1 · 173 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 1: Foundations of Algorithm Analysis & Asymptotic Notation > Asymptotic Notation Definitions:

```text
- Big-O (O) - Upper Bound: f(n) = O(g(n)) if there exist positive constants c and n0 such that f(n) ≤ c · g(n) for all n ≥ n0. Represents worst-case running time.

- Big-Omega (Ω) - Lower Bound: f(n) = Ω(g(n)) if f(n) ≥ c · g(n) for all n ≥ n0. Represents best-case running time.

- Big-Theta (Θ) - Tight Bound: f(n) = Θ(g(n)) if c1 · g(n) ≤ f(n) ≤ c2 · g(n). Represents exact growth rate.
```

## chunk 3 · p.1 · 179 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 1: Foundations of Algorithm Analysis & Asymptotic Notation > Common Asymptotic Growth Classes

```text
|Class Name|Notation|Example Operation / Algorithm|Scale behavior (n = 10<sup>6</sup>)|
|---|---|---|---|
|Constant|O(1)|Array indexing, Hash Map lookup|1 operation|
|Logarithmic|O(log n)|Binary Search, Balanced BST search|~20 operations|
|Linear|O(n)|Linear Search, Single pass traversals|1,000,000 operations|
|Linearithmic|O(n log n)|Merge Sort, Quick Sort (average case)|~20,000,000 operations|
|Quadratic|O(n<sup>2</sup>)|Bubble Sort, Nested loop iteration|10<sup>12</sup>operations|
```

## chunk 4 · p.1 · 93 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 1: Foundations of Algorithm Analysis & Asymptotic Notation > Common Asymptotic Growth Classes > Quiz Benchmark Questions:

```text
1. What is the formal mathematical condition required for a function f(n) to belong to Θ(g(n))?

2. If an algorithm processes n elements in 3n<sup>2</sup> + 15n + 100 steps, what is its tight Big-O bound?

Page 1 of 10

CS-201: Data Structures & Algorithms
```

## chunk 5 · p.2 · 119 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 2: Linear Data Structures - Arrays & Dynamic Vectors — **prompt sees only the first 500 of 524 chars** (cut marked ⟂)

```text
An Array is a contiguous block of memory allocated to store homogeneous elements. Because elements reside sequentially in physical memory, accessing any element by its index i occurs in constant time O(1) via the address formula: Address(A[i]) = BaseAddress + i × ElementSize. However, insertions and deletions at arbitrary positions require shifting elements, incurring an O(n) time complexity. Dynamic Arrays (such as C++ `std::vector` or Python `list`) overcome fixed-size constraints by automatic
⟂
ally resizing when full.
```

## chunk 6 · p.2 · 27 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 2: Linear Data Structures - Arrays & Dynamic Vectors > Dynamic Vector Amortized Insertion Analysis (C++)

```text
#include <iostream> template <typename T> class DynamicArray { private: T* data; size_t capacit
```

## chunk 7 · p.2 · 101 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 2: Linear Data Structures - Arrays & Dynamic Vectors > Dynamic Vector Amortized Insertion Analysis (C++) > Amortized Complexity Analysis of Doubling Strategy:

```text
When capacity doubles at powers of 2, inserting n elements causes copies at steps 1, 2, 4, 8, ..., 2<sup>k</sup> . Total copies = ∑i=0k 2i = 2k+1 - 1 < 2n. Averaged across all n insertions, each `push_back` costs O(2n / n) = O(1) amortized time.
```

## chunk 8 · p.2 · 106 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 2: Linear Data Structures - Arrays & Dynamic Vectors > Dynamic Vector Amortized Insertion Analysis (C++) > Quiz Benchmark Questions:

```text
1. Why does a dynamic array capacity doubling strategy yield O(1) amortized insertion, whereas increasing capacity by a fixed constant k yields O(n) amortized time?

2. Given a 0-indexed array with base address 0x1000 and 4-byte integers, calculate the memory location of `A[12]`.

Page 2 of 10

CS-201: Data Structures & Algorithms
```

## chunk 9 · p.3 · 88 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 3: Singly and Doubly Linked Lists

```text
A Linked List is a linear data structure where elements (nodes) are stored non-contiguously in memory. Each node contains data and a pointer (reference) to the next node. Unlike arrays, linked lists allow O(1) insertions and deletions at known pointer locations without shifting elements. However, random access is impossible; accessing the k-th element requires traversing nodes sequentially in O(k) time.
```

## chunk 10 · p.3 · 160 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 3: Singly and Doubly Linked Lists

```text
class Node: def __init__(self, data): self.data = data self.next = None class LinkedList: def _

|Operation|Array / Vector|Singly Linked List|Doubly Linked List|
|---|---|---|---|
|Random Access (i-th element)|O(1)|O(n)|O(n)|
|Insertion / Deletion at Head|O(n)|O(1)|O(1)|
|Insertion / Deletion at Tail|O(1)Amortized|O(n)(without tail ptr)|O(1)(with tail ptr)|
|Extra Memory Overhead|Minimal (unused capacity)|1 Pointer per node|2 Pointers per node|
```

## chunk 11 · p.3 · 121 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 3: Singly and Doubly Linked Lists > Comparison: Arrays vs. Linked Lists > Quiz Benchmark Questions:

```text
1. Trace the pointer updates during iterative singly linked list reversal. What are the values of `prev`, `current`, and `next_node` after 2 iterations on list `1 -> 2 -> 3 -> 4`?

2. How does Floyd's Cycle-Finding Algorithm (Tortoise and Hare) detect loops in a linked list in O(n) time and O(1) space?

Page 3 of 10

CS-201: Data Structures & Algorithms
```

## chunk 12 · p.4 · 255 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 4: Stacks, Queues, and Monotonic Variants — **prompt sees only the first 500 of 972 chars** (cut marked ⟂)

```text
Stacks follow a Last-In, First-Out (LIFO) discipline, while Queues operate on a First-In, First-Out (FIFO) basis. Both are abstract data types (ADTs) restricted to specific access points. Stacks support `push`, `pop`, and `peek` operations in O(1) time, making them essential for function call stacks, expression evaluation, and backtracking. Queues support `enqueue` and `dequeue` in O(1) time, operating as buffers for BFS traversals and process scheduling.

Monotonic Decreasing Stack for Next Gre
⟂
ater Element (C++)

#include <vector> #include <stack> std::vector<int> nextGreaterElement(const std::vector<int>&

Stack Applications: Infix to Postfix Conversion (Shunting-Yard Algorithm)

When parsing mathematical expressions like `(A + B) * C`, operators are pushed onto an operator stack according to precedence rules, while operands pass directly to output. The resulting postfix expression `A B + C *` eliminates parenthesis ambiguity during stack-based evaluation.
```

## chunk 13 · p.4 · 103 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 4: Stacks, Queues, and Monotonic Variants > Quiz Benchmark Questions:

```text
1. Evaluate the postfix expression `5 3 2 * + 8 2 / -` using a stack execution model. Show the stack state at each token.

2. Explain how to implement a Queue using two Stacks such that `enqueue` is O(1) and `dequeue` is O(1) amortized time.

Page 4 of 10

CS-201: Data Structures & Algorithms
```

## chunk 14 · p.5 · 101 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 5: Recursion & Divide-and-Conquer Paradigm

```text
Recursion is a programming technique where a function solves a problem by calling reduced instances of itself. Every recursive function must define one or more base cases (stopping conditions) and a recursive step that moves toward the base case. Unbounded recursion results in stack overflow errors due to call stack memory exhaustion. The Divide-and-Conquer paradigm breaks a problem into disjoint subproblems, recursively solves them, and combines their solutions.
```

## chunk 15 · p.5 · 232 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 5: Recursion & Divide-and-Conquer Paradigm > Binary Search Implementation & Recurrence Analysis (Python) — **prompt sees only the first 500 of 502 chars** (cut marked ⟂)

```text
def binary_search(arr, low, high, target): if low > high: return -1 # Base Case: Element not pr

The Master Theorem for Divide-and-Conquer Recurrences: For recurrences of the form T(n) = a T(n/b) + f(n), where a ≥ 1, b > 1: Case 1: If f(n) = O(n<sup>logba - ε</sup> ), then T(n) = Θ(n<sup>logba</sup> ). Case 2: If f(n) = Θ(n<sup>logba</sup> log<sup>k</sup> n), then T(n) = Θ(n<sup>logba</sup> log<sup>k+1</sup> n). Case 3: If f(n) = Ω(n<sup>logba + ε</sup> ) and regularity holds, then T(n) = Θ(f(n)
⟂
).
```

## chunk 16 · p.5 · 104 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 5: Recursion & Divide-and-Conquer Paradigm > Binary Search Implementation & Recurrence Analysis (Python) > Quiz Benchmark Questions:

```text
1. Apply the Master Theorem to solve T(n) = 2 T(n/2) + n. Identify values for a, b, and the matching case.

2. Why is `mid = low + (high - low) // 2` preferred over `mid = (low + high) // 2` in typed languages like C++ or Java?

Page 5 of 10

CS-201: Data Structures & Algorithms
```

## chunk 17 · p.6 · 109 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 6: Sorting Algorithms - Comparison & Non-Comparison

```text
Sorting arranges elements in a specified order. Comparison-based algorithms have a lower bound proof of Ω(n log n) time complexity. Quadratic algorithms (O(n<sup>2</sup> )) like Bubble, Selection, and Insertion Sort are inefficient for large inputs, whereas Merge Sort and Quick Sort achieve O(n log n). Non-comparison sorts (Counting Sort, Radix Sort) break the Ω(n log n) barrier by exploiting key distribution properties.
```

## chunk 18 · p.6 · 195 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 6: Sorting Algorithms - Comparison & Non-Comparison

```text
int partition(std::vector<int>& arr, int low, int high) { int pivot = arr[low + (high - low) /

|Algorithm|Best Time|Average Time|Worst Time|Space|Stable?|
|---|---|---|---|---|---|
|Insertion Sort|O(n)|O(n<sup>2</sup>)|O(n<sup>2</sup>)|O(1)|Yes|
|Merge Sort|O(n log n)|O(n log n)|O(n log n)|O(n)|Yes|
|Quick Sort|O(n log n)|O(n log n)|O(n<sup>2</sup>)|O(log n)|No|
|Counting Sort|O(n + k)|O(n + k)|O(n + k)|O(k)|Yes|
```

## chunk 19 · p.6 · 96 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 6: Sorting Algorithms - Comparison & Non-Comparison > Sorting Algorithms Comprehensive Complexity Matrix > Quiz Benchmark Questions:

```text
1. Why does Quick Sort degenerate to O(n<sup>2</sup> ) time complexity when bad pivot choices (e.g., sorted input with first element as pivot) occur?

2. What is meant by algorithmic "stability", and why is Merge Sort preferred over Quick Sort when stability is mandatory?

Page 6 of 10

CS-201: Data Structures & Algorithms
```

## chunk 20 · p.7 · 101 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 7: Trees & Binary Search Trees (BST)

```text
A Tree is a hierarchical non-linear data structure consisting of nodes connected by edges. A Binary Tree restricts each node to at most two children (left and right). A Binary Search Tree (BST) enforces the structural property: for any node X, all keys in the left subtree are strictly smaller than X.key, and all keys in the right subtree are strictly greater. Inorder traversal of a BST yields elements in sorted ascending order.
```

## chunk 21 · p.7 · 30 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 7: Trees & Binary Search Trees (BST) > BST Node Insertion and Inorder Traversal (Python)

```text
class BSTNode: def __init__(self, key): self.key = key self.left = None self.right = None def i
```

## chunk 22 · p.7 · 88 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 7: Trees & Binary Search Trees (BST) > BST Node Insertion and Inorder Traversal (Python) > Tree Traversal Taxonomies:

```text
- Pre-order (Root, Left, Right): Used for tree cloning and serialization.

- In-order (Left, Root, Right): Visits BST nodes in ascending sorted sequence.

- Post-order (Left, Right, Root): Used for node deletion and expression tree evaluation. Level-order (BFS): Explores nodes level-by-level using an auxiliary queue.
```

## chunk 23 · p.7 · 130 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 7: Trees & Binary Search Trees (BST) > BST Node Insertion and Inorder Traversal (Python) > Quiz Benchmark Questions:

```text
1. In a degenerate (skewed) BST created by inserting sorted numbers `1, 2, 3, 4, 5`, what are the worst-case search and insertion times?

2. Given Pre-order sequence `[F, B, A, D, C, E, G, I, H]` and In-order sequence `[A, B, C, D, E, F, G, H, I]`, reconstruct the tree.

Page 7 of 10

CS-201: Data Structures & Algorithms
```

## chunk 24 · p.8 · 138 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 8: Self-Balancing Trees (AVL) & Heaps

```text
To prevent BST degradation to linear linked lists (O(n) search), self-balancing trees enforce height invariants. An AVL Tree requires that for every node, the height difference between left and right subtrees (Balance Factor) is at most ±1: BF = Height(Left) - Height(Right) ∈ {-1, 0, 1}. Violations trigger rotations (LL, RR, LR, RL). A Binary Heap is a complete binary tree satisfying the heap property: Max-Heap (Parent ≥ Child) or Min-Heap (Parent ≤ Child).
```

## chunk 25 · p.8 · 27 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 8: Self-Balancing Trees (AVL) & Heaps > Binary Min-Heap Array Representation & Percolate Up (C++)

```text
#include <vector> #include <algorithm> class MinHeap { private: std::vector<int> heap; void hea
```

## chunk 26 · p.8 · 77 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 8: Self-Balancing Trees (AVL) & Heaps > Binary Min-Heap Array Representation & Percolate Up (C++) > AVL Tree Rotation Rules summary:

```text
- Single Left Rotation (RR Insertion): Applied when a right child's right subtree causes imbalance. Single Right Rotation (LL Insertion): Applied when a left child's left subtree causes imbalance. Double Rotation (LR / RL Insertion): Performed as a combination of sub-tree single rotation followed by root rotation.
```

## chunk 27 · p.8 · 101 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 8: Self-Balancing Trees (AVL) & Heaps > Binary Min-Heap Array Representation & Percolate Up (C++) > Quiz Benchmark Questions:

```text
1. Starting with an empty AVL tree, insert keys `10, 20, 30` in order. Show the resulting tree before and after the balancing rotation.

2. Why does building a binary heap from an unsorted array of n elements using `buildHeap` cost O(n) time instead of O(n log n)?

Page 8 of 10

CS-201: Data Structures & Algorithms
```

## chunk 28 · p.9 · 139 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 9: Hash Tables & Collision Resolution — **prompt sees only the first 500 of 548 chars** (cut marked ⟂)

```text
A Hash Table maps keys to values using a Hash Function h(k), computing array indices in average O(1) time. A good hash function distributes keys uniformly across slots. When two keys map to the same index (h(k1) = h(k2)), a Collision occurs. The two primary collision resolution techniques are Separate Chaining (linked lists per bucket) and Open Addressing (probing contiguous memory slots).

Hash Table with Separate Chaining Implementation (Python)

class HashTable: def __init__(self, capacity=11
⟂
): self.capacity = capacity self.table = [[] for
```

## chunk 29 · p.9 · 122 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 9: Hash Tables & Collision Resolution > Open Addressing Probing Techniques:

```text
- Linear Probing: h(k, i) = (h'(k) + i) mod m. Simple, but suffers from Primary Clustering.

- Quadratic Probing: h(k, i) = (h'(k) + c1i + c2i<sup>2</sup> ) mod m. Reduces primary clustering.

- Double Hashing: h(k, i) = (h1(k) + i · h2(k)) mod m. Eliminates secondary clustering.
```

## chunk 30 · p.9 · 119 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 9: Hash Tables & Collision Resolution > Quiz Benchmark Questions:

```text
1. Given a hash table of size 10 using h(k) = k mod 10 and Linear Probing, insert keys `12, 22, 32, 42`. State the final array index for each key.

2. What is the Load Factor (α) of a hash table, and at what threshold should resizing (rehashing) be triggered to maintain O(1) operations?

Page 9 of 10

CS-201: Data Structures & Algorithms
```

## chunk 31 · p.10 · 108 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 10: Graph Algorithms - BFS, DFS & Shortest Paths

```text
A Graph G = (V, E) consists of vertices (nodes) and edges (connections), which can be directed or undirected, weighted or unweighted. Graphs are represented using Adjacency Matrices (O(V<sup>2</sup> ) space, fast edge lookup) or Adjacency Lists (O(V + E) space, preferred for sparse graphs). Graph traversals underpin routing protocols, social network analysis, and topological sorting.
```

## chunk 32 · p.10 · 179 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 10: Graph Algorithms - BFS, DFS & Shortest Paths

```text
#include <vector> #include <queue> const int INF = 1e9; using pii = std::pair<int, int>; // {di

|Algorithm|Primary Use Case|Time Complexity|Space Complexity|
|---|---|---|---|
|Breadth-First Search (BFS)|Unweighted Shortest Path|O(V + E)|O(V)|
|Depth-First Search (DFS)|Cycle Detection / Topological Sort|O(V + E)|O(V)|
|Dijkstra's Algorithm|Weighted Shortest Path (Non-negative)|O((V + E) log V)|O(V)|
|Bellman-Ford Algorithm|Shortest Path with Negative Edges|O(V · E)|O(V)|
```

## chunk 33 · p.10 · 95 tokens · Data Structures & Algorithms: Comprehensive Course Notes > Chapter 10: Graph Algorithms - BFS, DFS & Shortest Paths > Graph Traversal & Shortest Path Complexity Matrix > Quiz Benchmark Questions:

```text
1. Why does Dijkstra's algorithm fail on graphs containing negative edge weights, and why does Bellman-Ford correctly handle or detect them?

2. Explain how Kahn's algorithm (indegree-based BFS) performs Topological Sorting on a Directed Acyclic Graph (DAG).

Page 10 of 10

CS-201: Data Structures & Algorithms
```
