
# Hyper-paremeters tuning 

The tuning algorithm was created and developed by Piotr Płoński. It is heuristic algorithm created from combination of:

- **not-so-random** approach
- and **hill-climbing**

The approach is **not-so-random** because each algorithm has a defined set of hyper-parameters that usually works. At first step from not so random parameters an initial set of models is drawn. Then the hill climbing approach is used to pick best performing algorithms and tune them.

For each algorithm used in the AutoML the early stopping is applied.

The ensemble algorithm was implemented based on [Caruana paper](http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf).
