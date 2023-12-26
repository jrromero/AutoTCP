# Automated machine learning for test case prioritisation
## Supplementary material 

### Authors
- José Raúl Romero (a,b - corresponding author)
- Aurora Ramírez (a,b)
- Ángel Fuentes-Almoguera (a)
- Carlos García-Martínez (a, b)

(a) Department of Computer Science and Numerical Analysis, University of Córdoba, 14071, Córdoba, Spain

(b) Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI), Córdoba, Spain

### Abstract

Test case prioritisation (TCP) involves ordering and selecting the most relevant test cases to verify that the current functionality of a software system remains unaffected by code changes. Recently, TCP has been addressed by machine learning (ML), predicting the failure probability of each test case. However, software engineers may struggle to identify and implement the most suitable predictive models for TCP. As new builds adapt the test suite being tested, the model performance may decline as these new builds are incorporated. In this study, we address these challenges by applying automated workflow composition and hyperparameter optimisation. Both are considered tasks within automated machine learning (AutoML). With this aim, our proposal employs grammar-guide genetic programming as the underlying mechanism for implementing the AutoML algorithm. Our experimental results demonstrate that our approach can adapt to the particularities of the system under test, selecting the most appropriate workflow and hyperparameters for each build. More importantly, our approach eliminates the need for testers to possess extensive ML knowledge, while enabling them to generate workflows suited to successive changes in SUT builds. This research showcases the potential of AutoML in software engineering, specifically for the TCP problem.

### Supplementary material

#### Datasets

We use the TCP dataset provided by Yaraghi et al. in their paper: ["Scalable and Accurate Test Case Prioritization in Continuous Integration Contexts"](https://doi.org/10.1109/TSE.2022.3184842). The dataset contains 25 open source Java projects, and can be downloaded from [Zenodo](https://zenodo.org/record/6415365#.Y9Kw43bMJD8).

#### Grammar

The directory [grammar](/grammar) includes the full specification of the context-free grammar used by our grammar-guided genetic programming algorithm to produce ML workflows. The reduced grammars applied in the experiment to address RQ1 are also included.

#### Results

The directory [results](/results) includes the detailed results obtained by our proposed method (AutoTCP), divided into three research questions:

- RQ1: How do the different configurations of AutoTCP affect its effectiveness in addressing the TCP problem?
- RQ2: Can AutoTCP generate distinct ML workflows adapted to different SUT characteristics and their evolution?
- RQ3: How does AutoTCP compare with state-of-the-art methods?

#### Statistical analysis

The directory [statistical analysis](statistical_analysis) includes the reports generated by the statistical analysis carried out for RQ1 and RQ3.
