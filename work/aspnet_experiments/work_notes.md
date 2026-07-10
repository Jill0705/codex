# Stage-1 Experiment Notes

## Hypothesis

Adaptive sub-prototype allocation should outperform both one-prototype-per-class and fixed-K multi-prototype classifiers under long-tailed class distributions.

## First Valid Result

A useful first result is not necessarily SOTA. The first result should show a clean ordering on CIFAR-100-LT IF=100:

CE baseline < single prototype <= fixed K < adaptive K

If this ordering fails, inspect:

- whether prototype temperature is too low or too high;
- whether fixed K overfits tail classes;
- whether adaptive K range is too narrow;
- whether backbone representation is too weak before prototype learning.

## What Not To Add Yet

Do not add EMA, Hungarian matching, or logit adjustment until the above comparison is understood.
