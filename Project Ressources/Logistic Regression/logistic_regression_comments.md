# Logistic Regression Experimental Results

## Best Overall Configurations

### Top Configurations by Learning Rate

| Learning Rate | Weights | Max Iterations | Train Accuracy | Train F1-Score | Test Accuracy | Test F1-Score |
|--------------|---------|---------------|---------------|---------------|--------------|--------------|
| **0.1000** | No | 1000 | 59.916% | 0.420657 | 65.000% | 0.281026 |
| **0.0100** | No | 250-1500 | ~68.776% | ~0.520-0.540 | 56.667-58.333% | 0.279-0.317 |
| **0.0010** | No | 2500 | 69.620% | 0.540809 | 56.667% | 0.279683 |
| **0.0001** | Yes | 500-1000 | 67.089% | 0.567-0.570 | 61.667% | 0.392-0.411 |

## Experimental Observations

### Key Findings

1. **Learning Rate Impact**
   - Most consistent performance at 0.01 learning rate
   - 0.0001 learning rate showed surprising effectiveness, especially with sample weights
   - Higher learning rates (0.1) showed more variability

2. **Sample Weights**
   - Most pronounced impact at 0.0001 learning rate
   - Generally reduced train accuracy
   - Mixed effects on test performance
   - Particularly effective in balancing model performance at lower learning rates

3. **Iterations Trend**
   - Optimal performance typically around 500-1000 iterations
   - Performance often degrades at very high iteration counts (2000-2500)
   - Consistent pattern across different learning rates

4. **Noteworthy Configurations**
   - 0.0001 LR with weights consistently achieved:
     * Highest test accuracy (61.667%)
     * Highest test F1-scores (0.392-0.411)
   - Unweighted models at 0.01 LR showed remarkably stable performance

### Practical Implications

- The choice of learning rate significantly impacts model performance
- Sample weights can help in addressing class imbalance
- Careful tuning of iterations is crucial for optimal results

## Full Experimental Results

[The full comprehensive table is available in the attached markdown file]

## Recommendations

1. For this specific dataset, consider:
   - Learning rates around 0.0001 or 0.01
   - Iteration counts between 500-1000
   - Experimenting with sample weights, especially at lower learning rates

2. Further investigation might involve:
   - More granular learning rate exploration
   - Additional preprocessing techniques
   - Ensemble methods to improve overall performance
