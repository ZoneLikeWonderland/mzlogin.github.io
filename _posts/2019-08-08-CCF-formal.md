---
layout: post
title: CCF 逻辑与形式化方法
categories: [course]
description: review 
keywords: CCF, logic, formal
---

## Temporal-Logic & Model Checking

- Kripke structure
  - **K=<S,R,L>**

### Symbolic Logic

- Natural languages:
  - ambiguous
  - paradox

### Algebraic Logic

### The Temporal Logic CTL*

- Path quantifiers
  - **A**: for every infinite path from this state ~~(a state in an arbitrary path)~~
  - **E**: there exists an infinite path from this state
- Temporal operators
  - **X**p: p holds at the next state
  - **F**p: p holds at some states in the future
  - **G**p: p holds at all states in the future
  - q**U**p: Fp, and q holds at all states until p holds

### The Temporal Logic CTL

- **A**,**E** and **E**,**F**,**G**,**U** must stand in pairs

### The Temporal Logic LTL (Linear-time Temporal Logic)

- no **E**
- only one path

------

### Algorithmic Challenge

- symmetry reduction
- on-the-fly state-space exploration
- partial-order reduction
- assume-guarantee reasoning



- Symbolic methods
  - BDD
- Abstraction

------

### Future of Model Checking

- synthesis (generate program from specification)
- generalize, model measuring
- AI ???





## Modeling software 

- assertions



- Transitions:

  - the enabling condition: a predict

  - the transformation: a multiple assignment

    > a>b -> (b,c):=(c,d)

- execution: sequence of states

### LTL

- ☐: box, forever, **G**
- ◊: diamond, eventually, **F**
- ⚪: nexttime, **X**

### Automata over finite words

- automaton
- non-deterministic
- deterministic

### Correctness condition

- We want to find a correctness condition for a model to satisfy a specification

  - Language of a model: L(Model)
  - Language of a specification: L(Spec)

  We need: L(Model) ⊆ L(Spec)

- Program executions ⊆ Sequences satisfying Specification ⊆ All sequences

## Explicit-State Model Checking

### Breadth-first Search

- X!y: add y as the last element of X
- X!!y: add y as the first element of X
- X?y: remove the first element of X as y

### Depth-first Search

### Nested Depth-First Search

- dfs from a reachable final state to check if a loop exists



### Bitstate Hashing



## Binary Decision Diagrams

- (X0 && ! X1) || (X0 && X1) == X0
- Boolean Function API
- A BDD represents a Boolean function as an acyclic directed graph
  - Nonterminal vertices labeled by Boolean variables
  - Leaf vertices labeled with the values 1 and 0

![](/images/blog/2019-08-08_2.jpg)






- Restrict(f,i,b):
  - R(x1 && x2, 1, True) == x2
- ...

![](/images/blog/2019-08-08_1.jpg)


~~*Sleeping*~~



## BDD-based symbolic model checking

- represent sets of state
![](/images/blog/2019-08-08_3.jpg)

- represent transition relation
  - p': p of next state

![](/images/blog/2019-08-08_5.jpg)
![](/images/blog/2019-08-08_4.jpg)

- BddImage & BddPreImage
![](/images/blog/草图.png)



