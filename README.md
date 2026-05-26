# Ant Colony Algorithm for Word Sense Disambiguation - Natural Language Processing

This project implements a biologically inspired **Ant Colony Optimization (ACO)** algorithm for **Word Sense Disambiguation (WSD)**.

The implementation is inspired by the paper:

> [*Ant Colony Algorithm for the Unsupervised Word Sense Disambiguation of Texts: Comparison and Evaluation*](https://aclanthology.org/C12-1146/)  
> by Didier SCHWAB, Jérôme GOULIAN, Andon TCHECHMEDJIEV and Hervé BLANCHON

## About the Algorithm

The algorithm models Word Sense Disambiguation as a collective ant colony behavior problem.

Each possible meaning of a word is represented as a nest node in a semantic graph built from the text and WordNet synsets. Artificial ants explore the graph, collect energy, deposit pheromones and propagate semantic odour vectors extracted from dictionary definitions and related concepts.

During exploration, ants dynamically create semantic bridges between compatible word senses using Extended Lesk similarity. Over multiple simulation cycles, pheromone reinforcement and collective ant behavior allow the system to converge toward the most semantically coherent interpretation of the text.

The final predicted senses are selected using majority voting across multiple independent executions of the algorithm.

## Results

The algorithm was evaluated on the **SemEval-2007 Task 7 English coarse-grained all-words dataset**.

The experiment was executed on the **entire dataset** using the ant colony model with the parameters described in the paper.

## Evaluation Metrics

```text
Total gold annotations: 2269
Total predicted senses: 2264
Correct predictions: 1774

Accuracy:  0.7836
Precision: 0.7836
Recall:    0.7818
F1-score:  0.7827
```

## Requirements

Install dependencies:

```bash
pip install nltk
```

Download NLTK resources automatically:

- punkt
- averaged_perceptron_tagger
- wordnet
- omw-1.4

## Running the Project

Run the algorithm:

```bash
python ant_colony_algorithm.py
```