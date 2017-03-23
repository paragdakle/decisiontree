# **ReadMe File:**

## Run the program using the following command:

```
python decisionTree.py <L> <K> <training_set_csv_path> <validation_set_csv_path> <test_set_csv_path> <to_print>
```

   where-  
   L: integer (used in the post-pruning algorithm - number of outer iterations)  
   K: integer (used in the post-pruning algorithm - random number generator limit for inner loop)  
   to-print:{yes,no}

## Sample Execution:

```
python decisionTree.py 35 35 data_sets1/training_set.csv data_sets1/validation_set.csv data_sets1/test_set.csv no
```

## Sample Output:

   Entropy Heuristic Test Results: 75.85 percent accuracy.  
   Variance Impurity Heuristic Test Results: 76.65 percent accuracy.  
   Entropy Heuristic Post Pruning Test Results: 76.75 percent accuracy.  
   Variance Impurity Heuristic Post Pruning Test Results: 77.05 percent accuracy.  
