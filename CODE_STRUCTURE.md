# Code Structure Overview

This repository follows a clean, modular layout intended to mirror real research software projects.

---

## 1. C++ Implementation (`cpp/sim.cpp`)

The C++ version contains:

- `initialize_positions()`
- `initialize_random_generator()`
- Lennard–Jones potential matrix construction
- Metropolis acceptance logic
- Random-scan and all-at-once move strategies
- Simulation class (`simulator`)
- Standalone `main()` executable

---

## 2. Python Implementation (`python/`)

### `simulation.py`
- Pure Python implementation of the LJ potential  
- Vectorized NumPy Metropolis sampler  
- Output of energies, positions, and acceptance statistics  

### `plot_results.py`
- Reads CSV output  
- Produces energy and acceptance plots  
- Saves images in `results/`

---

## 3. Report (`report/`)
Contains the assignment PDF describing:

- Physical model  
- Methods  
- Implementation  
- Results and discussion  

---

## 4. Results (`results/`)
Stores:

- CSV outputs from Python and/or C++  
- Plot images  
- Any additional analysis data  
