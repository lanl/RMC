# ASBS Reference Implementation

## Paper

Adjoint Schrödinger Bridge Sampler

arXiv:2506.22565

## Official repository

https://github.com/facebookresearch/adjoint_samplers

## Reference repository state

Commit: `a972860`
Branch: `main`

## RMC base

Repository: https://github.com/lanl/RMC
Branch: `develop`
Commit at porting start: `7ac94c3`

## RMC development branch

Local branch: `asbs-dev`
Based on: `origin/develop`

## Implementation modes

- `paper`: direct numerical realization of the main-paper ASBS algorithm, without practical solver features that are not part of the core algorithm.
- `paper_repo`: practical implementation corresponding as closely as possible to the procedure and settings used in the official ASBS repository and associated experiments.

## Purpose

This file records the exact external reference state used for the RMC ASBS port. The port should preserve a distinction between the mathematical algorithm described in the paper and additional practical machinery used by the released implementation.
