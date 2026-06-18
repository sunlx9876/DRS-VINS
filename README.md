# DRS-VINS

DRS-VINS is a stereo visual-inertial system for robust localization in dynamic scenes. This repository is a research-oriented implementation built on top of VINS-Fusion and adapted toward the DRS-VINS methodology, including structure-aware triangulation, adaptive dynamic-weighted residual modeling, and confidence-guided sliding-window optimization.

## Overview

The current codebase focuses on the DRS-VINS pipeline for:

- stereo feature tracking with FAST + LK optical flow
- structure-aware stereo triangulation
- dynamic-consistency weighting from optical-flow and disparity residuals
- confidence-guided sliding-window optimization
- optional loop closure based on the original VINS-Fusion loop fusion module


## Implemented DRS-VINS Modules

The repository currently maps the DRS-VINS method into the following code path:

- frontend tracking and dynamic observation construction
- structure-aware triangulation and feature quality evaluation
- weighted visual-inertial optimization and confidence screening
- weighted projection factors
- optional loop closure


## Build

Environment requirements:

- Ubuntu 18.04
- ROS1 Melodic
- Ceres Solver
- OpenCV
- `realsense2_camera` package if you want to run the D435i pipeline


## Relation to VINS-Fusion

This project is not an independent rewrite from scratch. It explicitly reuses and modifies parts of VINS-Fusion.

The DRS-VINS implementation in this repository inherits and extends VINS-Fusion components such as:

- estimator framework and sliding-window optimization
- IMU preintegration and marginalization structure
- stereo feature tracking and observation flow
- loop closure / pose graph infrastructure


## Copyright and Attribution

This repository contains modified code derived from VINS-Fusion.

Original VINS-Fusion project information:

- Project: VINS-Fusion
- Authors: Tong Qin, Shaozu Cao, Jie Pan, Peiliang Li, Shaojie Shen
- Organization: Aerial Robotics Group, Hong Kong University of Science and Technology
- Original license: GPLv3

To preserve attribution correctly:

- original copyright/license headers inherited from VINS-Fusion are kept in the source files where applicable
- this repository remains distributed under GPLv3, see [LICENCE]
- the present DRS-VINS code should be understood as a modified, research-oriented derivative based in part on VINS-Fusion

If you describe or publish results from this repository, you should cite both:

- the DRS-VINS paper or manuscript
- the original VINS-Fusion / VINS-Mono work that this implementation builds on

