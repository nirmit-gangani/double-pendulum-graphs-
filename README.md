# Analysis of Predictability Horizon in a Chaotic Double Pendulum

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)

## Overview

This project investigates the **limits of predictability** in chaotic systems using the **double pendulum** as a model. Although governed by fully deterministic equations, the double pendulum exhibits extreme sensitivity to initial conditions — a hallmark of chaos known as the "butterfly effect".

The simulation uses the **fourth-order Runge-Kutta (RK4)** method to integrate the equations of motion and quantifies chaotic behavior through **Lyapunov exponent calculation** and **predictability horizon estimation**.

## Features

- Interactive double pendulum simulation with user-defined parameters (mass, length, initial angles)
- Real-time visualization of:
  - Pendulum trajectory in Cartesian space
  - Angles and angular velocities over time
  - Horizontal positions over time
- Quantitative chaos analysis:
  - Phase-space separation between nearby trajectories
  - Estimation of the **largest Lyapunov exponent** (\(\lambda\))
  - Calculation of the **practical predictability horizon**
- Energy conservation validation for numerical accuracy

## Files

- `double_pendulum_chaos.py` — Main simulation script with full visualization and Lyapunov analysis
- 
## Requirements

```bash
numpy
matplotlib
scipy
