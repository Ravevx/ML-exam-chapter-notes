# Chapter 1 --- Models, Data, Learning Problems

> **Exam-ready README**
>
> This chapter is about one big idea:
>
> **We have data → choose a model → define how good it is → learn the
> model → hope it works on future data.**
>
> The notes below collect the concepts, formulas, and the confusing
> points we discussed.

------------------------------------------------------------------------

# 0. The Big Picture

A machine-learning problem can be viewed as:

``` text
DATA
  ↓
Choose a model space / hypothesis space
  ↓
Choose a loss function
  ↓
Measure empirical risk on training data
  ↓
Possibly add regularization
  ↓
Optimize / minimize the objective
  ↓
Get a learned model
  ↓
Evaluate on unseen/future data
  ↓
Care about generalization / true risk
```

The central distinction to remember:

-   **Training data:** what we actually observe.
-   **Empirical risk:** how badly the model performs on the observed
    training data.
-   **True/expected risk:** how badly the model is expected to perform
    on future data.
-   **Regularization:** a preference for certain models, usually
    simpler/smaller ones, to reduce overfitting.

------------------------------------------------------------------------

# 1. What Is a Model?

A **model** is a function that maps inputs to predictions.

We can write:

\[ `\hat `{=tex}y\_`\theta`{=tex}(x) \]

where:

-   \(x\) = input/features
-   \(y\) = true target/label
-   (`\hat `{=tex}y\_`\theta`{=tex}(x)) = model's prediction
-   (`\theta`{=tex}) = model parameters/weights

Example:

``` text
x = years of experience
y = salary

model:
salary = 10000 × experience
```

The parameters (`\theta`{=tex}) determine the exact behavior of the
model.

------------------------------------------------------------------------

# 2. Model Space / Hypothesis Space

## Definition

The **model space** is the collection of all models that we allow
ourselves to consider.

Think:

> **Model space = all candidate models I am willing to choose from.**

Example:

If our model space contains straight lines:

\[ y = ax+b \]

then these are possible models:

\[ y=2x \]

\[ y=5x+3 \]

\[ y=-x+10 \]

But:

\[ y=x\^2 \]

is not in that model space because it is not a straight line.

### Mental picture

``` text
MODEL SPACE
│
├── Model A
├── Model B
├── Model C
├── Model D
└── ...
```

------------------------------------------------------------------------

# 3. Why Do We Search for a Model?

We usually don't know which model is best beforehand.

So learning can be thought of as:

``` text
Many possible models
        ↓
      SEARCH
        ↓
Choose a good model
```

The search is an **optimization problem**.

We define some objective that tells us what "good" means, and then
search for the model that minimizes/maximizes that objective.

------------------------------------------------------------------------

# 4. Training Data

A training dataset consists of examples:

\[ D={(x_i,y_i)}\_{i=1}\^{n} \]

where:

-   (x_i) = input of example (i)
-   (y_i) = correct target/label
-   \(n\) = number of training examples

The model sees the training data and learns parameters (`\theta`{=tex}).

------------------------------------------------------------------------

# 5. Loss Function

A **loss function** measures how bad a prediction is for one example.

General notation:

\[ `\ell`{=tex}(`\hat `{=tex}y,y) \]

Meaning:

> How bad is prediction (`\hat `{=tex}y) when the true answer is (y)?

## 5.1 0/1 loss

For classification:

\[ `\ell`{=tex}\_{0/1}(`\hat `{=tex}y,y)=
```{=tex}
\begin{cases}
0 & \text{if } \hat y=y\\
1 & \text{if } \hat y\neq y
\end{cases}
```
\]

So:

``` text
Correct → loss = 0
Wrong   → loss = 1
```

Other problems use other loss functions.

For regression, for example, squared loss can be:

\[ `\ell`{=tex}(`\hat `{=tex}y,y)=(`\hat `{=tex}y-y)\^2 \]

For probabilistic classification, cross-entropy/log loss may be used.

### Important:

**There is not one universal loss function.**

The loss depends on the learning problem.

------------------------------------------------------------------------

# 6. Empirical Risk

This is one of the most important definitions in the chapter.

## Definition

**Empirical risk = average loss on the training data.**

\[ `\boxed{
R_{\mathrm{emp}}(\theta)
=
\frac{1}{n}
\sum_{i=1}^{n}
\ell(\hat y_\theta(x_i),y_i)
}`{=tex} \]

Read it as:

> "For every training example, calculate the loss, then take the
> average."

### With 0/1 loss

If we use 0/1 loss, empirical risk is simply:

> **fraction/percentage of training examples classified incorrectly.**

Example:

``` text
100 training examples
10 classified incorrectly

empirical risk = 10/100 = 0.10 = 10%
```

## Important confusion

Empirical risk does NOT have one numerical formula that always looks
identical.

The **general definition** is always:

\[ `\text{average loss on training data}`{=tex} \]

But the exact formula changes when the loss function changes.

For example:

### Squared loss:

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex}) = `\frac`{=tex}1n
`\sum`{=tex}*i (`\hat `{=tex}y*`\theta`{=tex}(x_i)-y_i)\^2 \]

### 0/1 loss:

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex}) = `\frac`{=tex}1n
`\sum`{=tex}*i
`\ell`{=tex}*{0/1}(`\hat `{=tex}y\_`\theta`{=tex}(x_i),y_i) \]

### Cross-entropy:

The same average-loss structure is used, but the loss inside the sum is
cross-entropy.

So:

> **Empirical risk is a concept; the exact formula depends on the chosen
> loss.**

------------------------------------------------------------------------

# 7. Empirical Risk = 0

If:

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex})=0 \]

then the model has zero training loss.

For 0/1 classification:

> The model correctly classifies every training example.

Example:

``` text
Actual       Prediction

Cat          Cat       ✓
Dog          Dog       ✓
Cat          Cat       ✓
Dog          Dog       ✓
```

Training error = 0.

## Important warning

Empirical risk = 0 does NOT mean the model is necessarily good.

It may have memorized the training data.

That is **overfitting**.

A model can have:

``` text
training error = 0%
future/test error = 30%
```

------------------------------------------------------------------------

# 8. Version Space

Suppose the model space contains many candidate models.

After seeing training data, some models make mistakes and some do not.

If we define consistency as **zero training error**, then:

> **Version space = all models in the model space that are perfectly
> consistent with the training data.**

So:

``` text
MODEL SPACE
│
├── A ✗
├── B ✓
├── C ✗
├── D ✓
├── E ✓
└── F ✗

VERSION SPACE
│
├── B ✓
├── D ✓
└── E ✓
```

Mathematically, under the zero-loss/consistency definition:

\[ `\mathcal `{=tex}V = {`\theta `{=tex}`\in `{=tex}`\mathcal `{=tex}H:
R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex})=0} \]

where (`\mathcal `{=tex}H) is the model/hypothesis space.

### Key idea

-   **Model space:** all allowed candidates.
-   **Version space:** candidates still consistent with the training
    data.
-   **Final learned model:** one model selected by the learning
    procedure.

------------------------------------------------------------------------

# 9. Supervised Learning

In supervised learning, we have labeled examples:

\[ (x_i,y_i) \]

The goal is to learn a mapping:

\[ x `\rightarrow `{=tex}y \]

There are several kinds of supervised learning problems.

------------------------------------------------------------------------

# 10. Supervised Learning: Classification

Classification predicts a discrete class/label.

Examples:

``` text
email → spam / not spam
image → cat / dog
patient → class A / class B
```

With 0/1 loss:

\[ `\ell`{=tex}\_{0/1}(`\hat `{=tex}y,y) =
```{=tex}
\begin{cases}
0 & \hat y=y\\
1 & \hat y\neq y
\end{cases}
```
\]

Common algorithms include:

-   Decision trees
-   Random forests
-   Logistic regression
-   k-nearest neighbors
-   Support vector machines
-   Neural networks

------------------------------------------------------------------------

# 11. Supervised Learning: Taxonomy Classification

A **taxonomy** is a hierarchy of classes.

Example:

``` text
Animal
├── Mammal
│   ├── Dog
│   └── Cat
└── Bird
    ├── Eagle
    └── Parrot
```

Taxonomy classification means assigning an instance to a class in such a
hierarchy.

The important distinction from "normal" classification is not that it
suddenly becomes unsupervised.

It is still **supervised learning** if labeled examples are provided.

The output structure is hierarchical.

Possible approaches include:

-   Flat classification
-   Hierarchical classification
-   Top-down classifiers
-   Decision trees
-   Neural networks designed for hierarchical labels

------------------------------------------------------------------------

# 12. Supervised Learning: Sequence and Structure Prediction

Here the output is not just one independent label.

The output has **structure**.

Examples:

### Sequence prediction

``` text
sentence → sequence of POS tags
audio → sequence of words
text → sequence of named entities
```

### Structured prediction

The output contains relationships/dependencies between components.

Examples:

-   Part-of-speech tagging
-   Named-entity recognition
-   Parsing
-   Speech recognition
-   Machine translation

The key idea:

> **The output variables are related rather than being independent
> labels.**

------------------------------------------------------------------------

# 13. Supervised Learning: Ranking

Ranking means learning to order items.

Example:

``` text
Search query: "best laptop"

Model output:

1. Laptop A
2. Laptop C
3. Laptop B
4. Laptop D
```

The exact numerical score may be less important than the ordering.

Applications:

-   Search engines
-   Information retrieval
-   Recommendation systems
-   Ranking documents
-   Ranking products

The training data may contain preferences such as:

\[ A \> B \]

meaning:

> A should rank higher than B.

------------------------------------------------------------------------

# 14. Supervised Learning: Recommendations

Recommendation predicts which items a user is likely to prefer.

Examples:

``` text
User → movies
User → products
User → songs
User → videos
```

Possible supervised targets:

-   Click/no click
-   Buy/no buy
-   Rating
-   Watch/no watch
-   Preference between two items

The system learns from historical user-item interactions.

Recommendation can involve classification, regression, ranking, or
combinations of them.

------------------------------------------------------------------------

# 15. Metric Learning

Metric learning is about learning a **distance/similarity function**.

The idea is:

> Instances that should be considered similar should be close in the
> learned representation/metric, while dissimilar instances should be
> far apart.

For example:

``` text
Same class:
A ●
  ● B
   ● C

Different class:
                         ● D
```

A common modern approach is to learn an **embedding function**:

\[ f(x) \]

such that:

\[ `\text{same class}`{=tex} `\Rightarrow`{=tex}
f(x_i)`\text{ and }`{=tex}f(x_j)`\text{ are close}`{=tex} \]

and:

\[ `\text{different class}`{=tex} `\Rightarrow`{=tex}
f(x_i)`\text{ and }`{=tex}f(x_j)`\text{ are far apart}`{=tex} \]

------------------------------------------------------------------------

# 16. Embeddings

An embedding converts an object into a vector.

Example:

``` text
Person A
   ↓
embedding model
   ↓
[0.2, 0.8, 0.1, 0.7]
```

The goal is for the vector space to have meaningful geometry.

If similar things are mapped to similar vectors:

``` text
same class → close
different class → far
```

then we can perform other tasks in the embedding space.

------------------------------------------------------------------------

# 17. Why Clustering Can Be Useful With Embeddings

Clustering is an **unsupervised** method that tries to discover groups.

A clustering algorithm does NOT magically know concepts such as:

``` text
tax slab
doctor
engineer
toxic
non-toxic
```

It only sees the representation/features and its clustering objective.

## Example

Suppose:

``` text
A salary = 10k
B salary = 20k
C salary = 30k
D salary = 32k
```

If we cluster using only salary, the algorithm sees distances such as:

``` text
10k ---- 20k ---- 30k -- 32k
 A        B        C      D
```

It has no idea that 32k is a tax boundary.

It might group C and D because they are close.

### Important lesson

> **Clustering discovers groups according to the similarity encoded in
> the input representation/objective.**

It does not know what groups humans care about unless that information
has somehow been encoded.

------------------------------------------------------------------------

# 18. Professor's Embedding + Clustering Idea

The professor's statement was essentially:

> If we have a trained embedding function where same-class instances
> have similar embeddings and different-class instances have different
> embeddings, then clustering can discover the groups in that embedding
> space.

Pipeline:

``` text
Original data
     ↓
Trained embedding function
     ↓
Meaningful vectors
     ↓
Clustering
     ↓
Groups
```

The embedding function provides a meaningful space.

The clustering algorithm then finds groups inside that space.

### Is the professor wrong?

No.

But the important subtlety is:

> **The clustering algorithm itself does not know the semantic meaning
> of the classes. The useful similarity structure must come from the
> representation/features, the clustering objective, or some other prior
> information.**

------------------------------------------------------------------------

# 19. Classification vs Clustering

## Classification

Usually supervised.

You have labels:

``` text
10k → Low
20k → Low
30k → Low
32k → High
```

The model learns the mapping.

## Clustering

Usually unsupervised.

You give:

``` text
10k
20k
30k
32k
```

and ask:

> Find groups.

It does not know what "Low tax" or "High tax" means.

### Key sentence

> **Classification learns predefined labels; clustering discovers groups
> according to similarity.**

------------------------------------------------------------------------

# 20. Tax-Slab Example

Suppose the actual rule is:

``` text
salary < 32k → Low tax
salary ≥ 32k → High tax
```

If you already know the rule, you don't need ML:

``` python
if salary < 32000:
    slab = "Low"
else:
    slab = "High"
```

If you have labeled examples, it can be a supervised classification
problem.

If you have no labels and just run clustering on salary, clustering does
not know the tax boundary.

------------------------------------------------------------------------

# 21. Uncertainty in Learning

We don't know the true model that generated the data.

There may be many models that explain the observed training data.

Therefore:

> **Learning involves uncertainty about which model is truly
> appropriate.**

This leads to the idea of learning as an optimization problem.

------------------------------------------------------------------------

# 22. Learning as an Optimization Problem

We have candidate models (`\theta`{=tex}).

We define an objective function.

Then:

\[ `\boxed{
\theta^*
=
\arg\min_\theta
\text{objective}(\theta)
}`{=tex} \]

Read:

> Choose the model parameters (`\theta`{=tex}) that give the smallest
> objective.

The objective may contain:

-   training loss
-   regularization
-   other constraints/preferences

------------------------------------------------------------------------

# 23. Regularizer

A regularizer is a function:

\[ `\Omega`{=tex}(`\theta`{=tex}) \]

It expresses a preference about the model.

The professor's interpretation:

> **(`\Omega`{=tex}(`\theta`{=tex})) expresses assumptions about whether
> model (`\theta`{=tex}) is a priori probable/preferred.**

And:

> **(`\Omega`{=tex}) is independent of the particular training
> examples.**

It looks at properties of the model itself.

Examples:

-   How many weights are non-zero?
-   How large are the weights?
-   How complex is the model?

### Very important correction

A regularizer does **not normally add noise**.

Some techniques such as dropout or noise injection can act as
regularization techniques, but L0/L1/L2 regularization work by adding a
penalty to the objective.

Think:

> **Regularizer = model preference / penalty.**

Not:

> "Regularizer = noise."

------------------------------------------------------------------------

# 24. Why Regularization Helps With Overfitting

Without regularization:

``` text
"Fit the training data as well as possible."
```

A highly flexible model may memorize training data.

With regularization:

``` text
"Fit the training data well,
BUT prefer certain model properties."
```

So we accept potentially slightly worse training performance in exchange
for a model that may generalize better.

This is a trade-off:

``` text
Training fit
     ↕
Model simplicity / preference
```

------------------------------------------------------------------------

# 25. L0 Regularization

L0 counts the number of non-zero parameters.

\[ `\boxed{
\Omega_0(\theta)
=
\#\{i:\theta_i\neq0\}
}`{=tex} \]

Example:

\[ `\theta`{=tex}=\[5,0,0,2,0\] \]

There are two non-zero weights:

\[ `\Omega`{=tex}\_0(`\theta`{=tex})=2 \]

Interpretation:

> **Prefer models that use fewer features/active parameters.**

L0 strongly represents a sparse-model preference.

------------------------------------------------------------------------

# 26. L1 Regularization

L1 is usually:

\[ `\boxed{
\Omega_1(\theta)
=
\sum_i|\theta_i|
}`{=tex} \]

Example:

\[ `\theta`{=tex}=\[5,0,0,2,0\] \]

Then:

\[ `\Omega`{=tex}\_1(`\theta`{=tex}) = \|5\|+0+0+\|2\|+0 = 7 \]

L1 tends to encourage **sparsity**, meaning many weights can become
exactly zero.

This makes L1 useful for feature selection.

------------------------------------------------------------------------

# 27. L2 Regularization

L2 is commonly:

\[ `\boxed{
\Omega_2(\theta)
=
\sum_i\theta_i^2
}`{=tex} \]

Example:

\[ `\theta`{=tex}=\[5,0,0,2,0\] \]

Then:

\[ `\Omega`{=tex}\_2(`\theta`{=tex}) = 5^2+2^2 = 25+4 = 29 \]

L2 strongly penalizes large weights and generally encourages weights to
be small rather than necessarily exactly zero.

------------------------------------------------------------------------

# 28. L0 vs L1 vs L2 --- Quick Table

  Regularizer   Formula                                 Main preference
  ------------- --------------------------------------- ---------------------
  L0            number of non-zero weights              Few active features
  L1            (`\sum`{=tex}\_i                        `\theta`{=tex}\_i
  L2            (`\sum`{=tex}\_i`\theta`{=tex}\_i\^2)   Small weights

### Easy memory trick

``` text
L0 → "How many weights are ON?"
L1 → "How much total weight?"
L2 → "How much squared weight?"
```

------------------------------------------------------------------------

# 29. Professor's Candidate-Model Example

Suppose there are three candidate parameter vectors:

\[ `\theta`{=tex}\_1=\[1,0,0,0,0,0,1\] \]

\[ `\theta`{=tex}\_2=\[2,0,1,0,0,1,0\] \]

\[ `\theta`{=tex}\_3=\[1,0,0,0,2,0,1\] \]

## L0

Count non-zero weights:

### (`\theta`{=tex}\_1)

Two non-zero values:

\[ `\Omega`{=tex}\_0(`\theta`{=tex}\_1)=2 \]

### (`\theta`{=tex}\_2)

Three non-zero values:

\[ `\Omega`{=tex}\_0(`\theta`{=tex}\_2)=3 \]

### (`\theta`{=tex}\_3)

Three non-zero values:

\[ `\Omega`{=tex}\_0(`\theta`{=tex}\_3)=3 \]

------------------------------------------------------------------------

## L1

### (`\theta`{=tex}\_1)

\[ 1+1=2 \]

\[ `\Omega`{=tex}\_1(`\theta`{=tex}\_1)=2 \]

### (`\theta`{=tex}\_2)

\[ \|2\|+\|1\|+\|1\|=4 \]

\[ `\Omega`{=tex}\_1(`\theta`{=tex}\_2)=4 \]

### (`\theta`{=tex}\_3)

\[ \|1\|+\|2\|+\|1\|=4 \]

\[ `\Omega`{=tex}\_1(`\theta`{=tex}\_3)=4 \]

------------------------------------------------------------------------

## L2

### (`\theta`{=tex}\_1)

\[ 1^2+1^2=2 \]

\[ `\Omega`{=tex}\_2(`\theta`{=tex}\_1)=2 \]

### (`\theta`{=tex}\_2)

\[ 2^2+1^2+1\^2=4+1+1=6 \]

\[ `\Omega`{=tex}\_2(`\theta`{=tex}\_2)=6 \]

### (`\theta`{=tex}\_3)

\[ 1^2+2^2+1\^2=1+4+1=6 \]

\[ `\Omega`{=tex}\_2(`\theta`{=tex}\_3)=6 \]

### Final table

  Model                   L0   L1   L2
  --------------------- ---- ---- ----
  (`\theta`{=tex}\_1)      2    2    2
  (`\theta`{=tex}\_2)      3    4    6
  (`\theta`{=tex}\_3)      3    4    6

Interpretation:

-   L0 prefers fewer non-zero weights.
-   L1 penalizes total absolute weight.
-   L2 penalizes large weights especially strongly.

------------------------------------------------------------------------

# 30. Regularized Empirical Risk

Now combine empirical risk and regularization.

The formula is:

\[ `\boxed{
R_{\mathrm{reg}}(\theta)
=
\frac1n
\sum_{i=1}^{n}
\ell(\hat y_\theta(x_i),y_i)
+
\lambda\Omega(\theta)
}`{=tex} \]

This has two parts:

\[ `\underbrace{
\frac1n\sum_i
\ell(\hat y_\theta(x_i),y_i)
}`{=tex}*{`\text{Empirical risk / training error}`{=tex}} +
`\underbrace{
\lambda\Omega(\theta)
}`{=tex}*{`\text{Regularization penalty}`{=tex}} \]

### Read it as:

> **Fit the training data well + prefer certain kinds of models.**

------------------------------------------------------------------------

# 31. What Is (`\lambda`{=tex})?

\[ `\lambda`{=tex} \]

controls how strongly we care about the regularizer.

### (`\lambda=0`{=tex})

\[ R\_{`\mathrm{reg}`{=tex}}=R\_{`\mathrm{emp}`{=tex}} \]

No regularization.

### Larger (`\lambda`{=tex})

The model is penalized more strongly according to
(`\Omega`{=tex}(`\theta`{=tex})).

If (`\lambda`{=tex}) is too large, the model may become too constrained
and underfit.

So (`\lambda`{=tex}) controls the trade-off between:

``` text
fit training data
       VS
prefer regularized/simple model
```

------------------------------------------------------------------------

# 32. Regularized Empirical Risk Minimization (RERM)

RERM means:

> **Choose the model that minimizes regularized empirical risk.**

Mathematically:

\[ `\boxed{
\theta^*
=
\arg\min_\theta
\left[
\frac1n
\sum_i
\ell(\hat y_\theta(x_i),y_i)
+
\lambda\Omega(\theta)
\right]
}`{=tex} \]

This is the learning procedure.

### Example from the lecture

If:

\[ `\ell`{=tex}=`\ell`{=tex}\_{0/1} \]

and:

\[ `\lambda=0.1`{=tex} \]

and L0 regularization is used:

\[ `\boxed{
\frac1n
\sum_i
\ell_{0/1}(\hat y_\theta(x_i),y_i)
+
0.1\Omega_0(\theta)
}`{=tex} \]

The candidate model with the smallest value is selected.

If the selected model is:

\[ `\hat `{=tex}y\_`\theta`{=tex}(x)=
```{=tex}
\begin{cases}
\text{toxic} & x_6\ge1\\
\text{nontoxic} & \text{otherwise}
\end{cases}
```
\]

then that model had the lowest regularized empirical risk among the
candidates considered.

------------------------------------------------------------------------

# 33. Risk vs Empirical Risk

This is another major distinction.

## Empirical risk

Uses the finite training dataset:

\[ `\boxed{
R_{\mathrm{emp}}(\theta)
=
\frac1n
\sum_i
\ell(\hat y_\theta(x_i),y_i)
}`{=tex} \]

Question:

> **How well does the model perform on the training data?**

------------------------------------------------------------------------

## True/Expected Risk

We actually care about future data.

Future examples are assumed to come from some unknown real-world
probability distribution:

\[ p(x,y) \]

The true/expected risk is:

\[ `\boxed{
R(\theta)
=
\mathbb E_{(X,Y)\sim p}
[
\ell(\hat y_\theta(X),Y)
]
}`{=tex} \]

Question:

> **How well will the model perform on future examples drawn from the
> real distribution?**

------------------------------------------------------------------------

# 34. Understanding the Integral Formula for Risk

The professor may write:

\[ R(`\theta`{=tex}) = `\sum`{=tex}*y `\int`{=tex}
`\ell`{=tex}(`\hat `{=tex}y*`\theta`{=tex}(x),y) p(x,y) ,dx \]

This is just the mathematical expansion of expected loss.

Break it apart:

### (`\ell`{=tex}(`\hat `{=tex}y\_`\theta`{=tex}(x),y))

How much loss do we get for this possible example?

### (p(x,y))

How likely is this particular input/label combination?

### Multiply them

\[ `\ell`{=tex}(`\hat `{=tex}y\_`\theta`{=tex}(x),y)p(x,y) \]

This gives:

> loss × probability of that situation.

### Integrate/sum over everything

We consider all possible inputs and labels.

So the entire formula means:

> **Take the loss for every possible future situation, weight it by how
> likely that situation is, and average it.**

------------------------------------------------------------------------

# 35. Discrete Version of Expected Risk

If there are only finitely many possible cases, think:

\[ R(`\theta`{=tex}) = `\sum`{=tex}*{x,y}
`\ell`{=tex}(`\hat `{=tex}y*`\theta`{=tex}(x),y)p(x,y) \]

This is simply a weighted average.

Example:

``` text
Situation A:
loss = 0, probability = 0.5

Situation B:
loss = 1, probability = 0.3

Situation C:
loss = 0, probability = 0.2
```

Expected risk:

\[ 0(0.5)+1(0.3)+0(0.2)=0.3 \]

So expected risk = 0.3.

------------------------------------------------------------------------

# 36. Why Can't We Usually Calculate True Risk Directly?

Because:

\[ p(x,y) \]

is unknown.

We don't know the exact probability distribution that generates all
future data.

We only have a finite sample.

So:

``` text
True distribution p(x,y)
        ↓
      unknown
        ↓
Can't directly calculate exact future risk
```

Instead, we use the training sample to estimate performance.

------------------------------------------------------------------------

# 37. Is Empirical Risk a Useful Estimator of True Risk?

This is the important question:

> **Can training error tell us how well the model will perform in the
> future?**

### Ideally, yes.

If:

-   training examples are representative of the real distribution,
-   examples are appropriately sampled,
-   there is enough data,
-   the model isn't badly overfitting,

then:

\[ `\boxed{
R_{\mathrm{emp}}(\theta)
\approx
R(\theta)
}`{=tex} \]

So empirical risk can be a useful estimator of true risk.

------------------------------------------------------------------------

# 38. Why Empirical Risk Can Fail

Suppose:

``` text
Training data:
10 examples
```

A very powerful model memorizes all 10.

Then:

\[ R\_{`\mathrm{emp}`{=tex}}=0 \]

But on future data:

\[ R(`\theta`{=tex})\>0 \]

possibly much larger.

This is overfitting.

### Therefore:

> **Low empirical risk does not guarantee low true risk.**

The real goal of ML is **generalization**.

------------------------------------------------------------------------

# 39. Generalization

Generalization means:

> **The model performs well on unseen data from the same underlying
> distribution.**

We want:

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex}) \]

to be a useful approximation to:

\[ R(`\theta`{=tex}) \]

rather than merely memorizing the training set.

### Good model

``` text
Training error: low
Future error: low
```

### Overfitted model

``` text
Training error: extremely low
Future error: high
```

### Underfitted model

``` text
Training error: high
Future error: high
```

------------------------------------------------------------------------

# 40. Why Regularization Connects to Generalization

Regularization does not directly calculate future performance.

Instead, it changes the learning preference:

``` text
Without regularization:
"Find a model that fits training data."

With regularization:
"Find a model that fits training data
and has preferred properties."
```

This can discourage overly complex/unstable models and often improve
generalization.

------------------------------------------------------------------------

# 41. Important Terminology Map

This is the section to read before the exam.

## Loss

**One example:**

\[ `\ell`{=tex}(`\hat `{=tex}y,y) \]

> How wrong is this prediction?

------------------------------------------------------------------------

## Empirical Risk

**Average loss over training data:**

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex}) = `\frac`{=tex}1n
`\sum`{=tex}*i `\ell`{=tex}(`\hat `{=tex}y*`\theta`{=tex}(x_i),y_i) \]

> How wrong is the model on average on the training set?

------------------------------------------------------------------------

## Regularizer

\[ `\Omega`{=tex}(`\theta`{=tex}) \]

> How much do we penalize/dislike this model according to our
> prior/model preference?

------------------------------------------------------------------------

## Regularized Empirical Risk

\[
R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex})+`\lambda`{=tex}`\Omega`{=tex}(`\theta`{=tex})
\]

> Training error + model penalty.

------------------------------------------------------------------------

## RERM

\[ `\theta`{=tex}\^\* = `\arg`{=tex}`\min`{=tex}\_`\theta
[
R_{\mathrm{emp}}(\theta)+\lambda\Omega(\theta)
]`{=tex}\]

> Choose the model with the smallest regularized empirical risk.

------------------------------------------------------------------------

## True/Expected Risk

\[ R(`\theta`{=tex}) = `\mathbb `{=tex}E\_{(X,Y)`\sim `{=tex}p} \[
`\ell`{=tex}(`\hat `{=tex}y\_`\theta`{=tex}(X),Y)\] \]

> Expected loss on future data.

------------------------------------------------------------------------

# 42. "Why Am I Seeing So Many Different Risk Formulas?"

This was one of the biggest confusions in this chapter.

Remember:

### There is a general definition:

\[ `\boxed{
\text{Empirical Risk = average training loss}
}`{=tex} \]

But the loss can change.

Therefore the exact formula changes.

Example:

``` text
0/1 loss
    ↓
empirical classification error

squared loss
    ↓
mean squared error

cross-entropy
    ↓
average cross-entropy
```

Then if regularization is added:

``` text
empirical risk
      +
λ × regularizer
      ↓
regularized empirical risk
```

And when we talk about future data:

``` text
expected/true risk
=
expected future loss
```

So don't memorize unrelated formulas.

Instead identify the layer:

``` text
ONE EXAMPLE
    ↓
LOSS

ALL TRAINING EXAMPLES
    ↓
EMPIRICAL RISK

EMPIRICAL RISK + MODEL PENALTY
    ↓
REGULARIZED EMPIRICAL RISK

MINIMIZE REGULARIZED EMPIRICAL RISK
    ↓
RERM

ALL POSSIBLE FUTURE DATA
    ↓
TRUE / EXPECTED RISK
```

------------------------------------------------------------------------

# 43. The Complete Learning Pipeline

Put everything together:

``` text
                    DATA
                      │
                      ↓
              Training examples
               (xᵢ, yᵢ)
                      │
                      ↓
                MODEL SPACE
             possible models θ
                      │
                      ↓
              Choose a LOSS
                      │
                      ↓
              Empirical Risk
       average loss on training data
                      │
                      ↓
             Add REGULARIZER
             λ Ω(θ)
                      │
                      ↓
       Regularized Empirical Risk
                      │
                      ↓
                    RERM
          minimize the objective
                      │
                      ↓
               Learned model
                      │
                      ↓
              Unseen future data
                      │
                      ↓
               TRUE / EXPECTED
                    RISK
                      │
                      ↓
               GENERALIZATION
```

------------------------------------------------------------------------

# 44. Exam Questions You Should Be Able to Answer

## Q: What is a model space?

**Answer:**

The set of candidate models/hypotheses that the learning algorithm is
allowed to choose from.

------------------------------------------------------------------------

## Q: Why search the model space?

Because we don't know beforehand which candidate model will best solve
the problem, so learning searches for a model that minimizes an
objective.

------------------------------------------------------------------------

## Q: What is empirical risk?

Average loss of a model over the training dataset:

\[ R\_{`\mathrm{emp}`{=tex}}(`\theta`{=tex}) = `\frac`{=tex}1n
`\sum`{=tex}*i `\ell`{=tex}(`\hat `{=tex}y*`\theta`{=tex}(x_i),y_i) \]

------------------------------------------------------------------------

## Q: What is the version space?

The set of models in the model space that are consistent with the
training data; under zero-one loss, commonly the models with empirical
risk 0.

------------------------------------------------------------------------

## Q: What is a regularizer?

A function (`\Omega`{=tex}(`\theta`{=tex})) that expresses a
preference/penalty for certain model properties, often favoring simpler
or smaller models.

------------------------------------------------------------------------

## Q: Does a regularizer add noise?

**Not in general.**

L0/L1/L2 regularization add a penalty to the objective. Noise-based
methods such as dropout/noise injection are separate regularization
techniques.

------------------------------------------------------------------------

## Q: What is L0?

Number of non-zero weights.

\[ `\Omega`{=tex}\_0(`\theta`{=tex})=#{i:`\theta`{=tex}\_i`\neq0`{=tex}}
\]

------------------------------------------------------------------------

## Q: What is L1?

Sum of absolute weights.

\[
`\Omega`{=tex}\_1(`\theta`{=tex})=`\sum`{=tex}\_i\|`\theta`{=tex}\_i\|
\]

Encourages sparse models.

------------------------------------------------------------------------

## Q: What is L2?

Sum of squared weights.

\[ `\Omega`{=tex}\_2(`\theta`{=tex})=`\sum`{=tex}\_i`\theta`{=tex}\_i\^2
\]

Encourages small weights.

------------------------------------------------------------------------

## Q: What is regularized empirical risk?

\[
`\frac`{=tex}1n`\sum`{=tex}*i`\ell`{=tex}(`\hat `{=tex}y*`\theta`{=tex}(x_i),y_i) +
`\lambda`{=tex}`\Omega`{=tex}(`\theta`{=tex}) \]

It balances fitting the training data with a preference for certain
model properties.

------------------------------------------------------------------------

## Q: What is RERM?

Regularized Empirical Risk Minimization:

\[ `\theta`{=tex}\^\* = `\arg`{=tex}`\min`{=tex}\_`\theta`{=tex} `\left[
\frac1n\sum_i\ell(\hat y_\theta(x_i),y_i)
+
\lambda\Omega(\theta)
\right]`{=tex}\]

Choose the model that minimizes regularized empirical risk.

------------------------------------------------------------------------

## Q: What is true/expected risk?

Expected loss on future examples from the unknown data-generating
distribution:

\[ R(`\theta`{=tex}) = `\mathbb `{=tex}E\_{(X,Y)`\sim `{=tex}p} \[
`\ell`{=tex}(`\hat `{=tex}y\_`\theta`{=tex}(X),Y)\] \]

------------------------------------------------------------------------

## Q: Is empirical risk the same as true risk?

No.

-   Empirical risk = training data.
-   True risk = expected future-data performance.

Empirical risk can estimate true risk when the training sample is
representative and sufficiently large, but overfitting can make training
risk misleading.

------------------------------------------------------------------------

# 45. Common Confusions --- Final Cheat Sheet

### Confusion 1:

"Empirical risk has one fixed formula."

**Correction:**

The definition is fixed:

\[ `\text{average loss on training data}`{=tex} \]

but the exact formula depends on the chosen loss.

------------------------------------------------------------------------

### Confusion 2:

"Regularizer adds noise."

**Correction:**

A regularizer usually adds a penalty:

\[ `\lambda`{=tex}`\Omega`{=tex}(`\theta`{=tex}) \]

Noise injection/dropout are other techniques that can act as
regularization.

------------------------------------------------------------------------

### Confusion 3:

"Regularization is unrelated to version space."

**Correction:**

They are connected conceptually.

Version space says:

> Which models fit the training data perfectly?

Regularization adds a preference among candidate models:

> Which kinds of models do we prefer?

------------------------------------------------------------------------

### Confusion 4:

"If empirical risk = 0, the model is perfect."

**Correction:**

No.

It means perfect performance on the training set according to that loss.

It may still overfit.

------------------------------------------------------------------------

### Confusion 5:

"Clustering knows what groups I want."

**Correction:**

No.

Clustering uses the similarity structure available in the
representation/features and its objective.

If you want tax slabs, the algorithm needs information that makes those
slabs meaningful.

------------------------------------------------------------------------

### Confusion 6:

"Embedding and clustering are the same thing."

**Correction:**

No.

``` text
Embedding → creates a meaningful representation
Clustering → discovers groups in that representation
```

------------------------------------------------------------------------

### Confusion 7:

"True risk is just another training-error formula."

**Correction:**

No.

Empirical risk uses observed training examples.

True risk is an expectation over the unknown future data distribution.

------------------------------------------------------------------------

# 46. Ultra-Short Exam-Day Summary

If you have only 2 minutes before the exam, remember this:

``` text
MODEL
= function making predictions

MODEL SPACE
= all candidate models

LOSS
= error on one example

EMPIRICAL RISK
= average loss on training data

VERSION SPACE
= models consistent with training data
  (commonly empirical risk = 0)

REGULARIZER Ω(θ)
= preference/penalty on model properties

L0
= number of non-zero weights

L1
= sum of absolute weights

L2
= sum of squared weights

REGULARIZED EMPIRICAL RISK
= empirical risk + λΩ(θ)

RERM
= minimize regularized empirical risk

TRUE/EXPECTED RISK
= expected loss on future data

GENERALIZATION
= performing well on unseen data

OVERFITTING
= training performance very good,
  future performance poor
```

The most important formula chain is:

\[ `\boxed{
\text{Loss}
\rightarrow
\text{Empirical Risk}
\rightarrow
\text{Regularized Empirical Risk}
\rightarrow
\text{RERM}
\rightarrow
\text{Generalization / True Risk}
}`{=tex} \]

And the most important conceptual chain is:

\[ `\boxed{
\text{Fit the data}
+
\text{Prefer good/simple models}
\rightarrow
\text{hope for good future performance}
}`{=tex} \]
