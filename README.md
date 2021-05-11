# Genetic Algorithms for Dimensionality Reduction of Hyperspectral Images

We tried implementing a Genetic search based Algorithm for feature selection of Hyperspectral Images that could be further used for the dimensionality reduction in hyperspectral images.

## GS Functions Used
- Selection
We do rank selection i.e. basically we preserve the top 60 (totalsamples)members of the population from the entire population.

- Crossover
In crossover, to preserve number of 1s (number of bands) we compute bands where x has 1 y has 0 and where x has 0 and y has 1 and swap equal number of positions that number of positions is decided by pswap.

- Mutation
In mutation, we flip a toss randomly, if we get true (or 1), we exchange 0s with 1 and 1s with 0 to obtain mutated species.ie, we perform swap mutation.

- Fitness Calculation based on SAM Classifier Accuracy, Mean Distance and Class Correlation
This fit function is used to calculate the fitness score by calling the score function and then sorting on the basis of highest fitness value.  The fitness function that is used is given below

## Running Instructions

Install CUDA toolkit(if not then run on google colab notebook)

Open the terminal and enter the code directory to run the following commands:
`pip install requirements.txt`
`python gsa.py`


The data in data/ folder will be loaded and graphs(confusion matrix) will be stored in data/ along with the kappa coefficients printed on terminal.

The compressed image will be stored as data/compressed_results.npy and the test information as other .npy files

## Results 
### Slow GSA
The Kappa coefficients for slow GSA are as followed,
- Split = 0.2
  - Full bands case = 0.44598538075694905
  - Compressed bands case = 0.44813983817964975
  - Improvement = 0.5%
- Split = 0.1
  - Full bands case = 0.44564570160557926
  - Compressed bands case = 0.4614073861985174
  - Improvement = 3.5%

### Fast GSA
The Kappa coefficients for fast GSA are as followed,
- Split = 0.2
  - Full bands case = 0.44598538075694905
  - Compressed bands case =0.5163158418049946
  - Improvement = 15.8%
 - Split = 0.1
  - Full bands case = 0.44564570160557926
  - Compressed bands case = 0.5143856294057647
  - Improvement = 15.4%


## Observations
At lower splits (with higher data) slow GSA performs better as it shows better performance at lower splits. Similarly, at higher splits (with lower data) fast GSA performs better. From this we observe that when we have large data to use Fast GSA and when we have small amount of data to use slow GSA

## Contributed by
- M V Girish - @girish-srivastav
- Liza Dahiya - @liza23
- Amey Anjalaker - @ameyanjalkar
- Bhavesh Patil
