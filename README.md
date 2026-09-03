# CMC Project 2 — Salamander Robot Locomotion

## Description

This project implements and studies a **Central Pattern Generator (CPG)** network for controlling locomotion in a simulated salamander robot (Polymander). The CPG is a biologically-inspired oscillator network that generates rhythmic motor commands for swimming and walking gaits without sensory feedback.

For more information on CPG theory, see the [reference paper](Report/science.1138353.pdf). This project was part of the **Computational Motor Control (CMC)** course at EPFL.

The work is divided into four exercises:

- **Exercise 1** — Implement and analyse the CPG network in isolation (no physics simulation): frequency, phase lag, amplitude gradients, interlimb/intralimb coupling.
- **Exercise 2** — Couple the CPG to a MuJoCo salamander model and produce swimming and walking behaviours with a ramp drive.
- **Exercise 3** — Study limb–spine coordination during walking; compare coupled vs. decoupled conditions and sweep key parameters (phase offsets, coupling weights).
- **Exercise 4** — Extended parameter sweeps and analysis (forward speed, cost of transport, lateral deviation).


## Installation

1. Clone the repository:

```bash
git clone https://github.com/martiiiina/cpg-salamander-controller.git
cd cpg-salamander-controller
```

2. Install the course simulation pack. The simulation framework (`farms_core`, `salamandra_simulation`, MuJoCo models) and project specification were provided by the **Computational Motor Control** course at EPFL (Prof. Ijspeert group). For installation instructions, see the [project installation guide](https://gitlab.com/farmsim/courses/cmc-2026-students/-/tree/main/project_installation?ref_type=heads).


3. Install the remaining Python dependencies:

```bash
pip install -r Python/project_requirements.txt
```

## Usage

All scripts must be run from the `Python/` directory.

```bash
cd Python
```

**Run a specific exercise:**

```bash
python exercise_all.py 1a          # CPG network analysis (no MuJoCo)
python exercise_all.py 2a          # Swimming simulation
python exercise_all.py 2b          # Walking simulation with ramp drive
python exercise_all.py 3.2         # Walking with disabled limb–spine coupling
python exercise_all.py 3.3         # Phase offset sweep
python exercise_all.py 3.4         # Coupling weight sweep
```

**Run all exercises sequentially:**

```bash
python project2.py
```

Simulation logs and figures are saved under `Python/logs/<exercise_tag>/`. Pre-computed results for select exercises are also available under `logs/`.


## Project structure

```
Python/
├── network.py                  # CPG network (SalamandraNetwork)
├── robot_parameters.py         # Robot & CPG parameters
├── simulation_parameters.py    # SimulationParameters dataclass
├── exercise_p1.py              # Ex 1 — network without MuJoCo
├── exercise_p2.py              # Ex 2 — swimming & walking
├── exercise_p3.py              # Ex 3 — limb–spine coordination
├── exercise_p3_analysis.py     # Ex 3 — plotting only, loads saved logs to produce figures
├── exercise_p4.py              # Ex 4 — extended sweeps
├── exercise_p4_analysis.py     # Ex 4 — plotting only, loads saved logs to produce figures
├── exercise_all.py             # Dispatcher for all exercises
├── project2.py                 # Main entry point
├── plot_results.py             # Aggregate plotting
├── plot_traj.py                # Trajectory visualisation
└── utils.py                    # Shared metrics (FWS, CoT, etc.)
```

## Report

For a detailed analysis of the results and findings from this project, see the [project report](Report/Report.pdf).

## Authors and acknowledgment

- **Martina Baroffio** — EPFL Neuro-X MSc
- **Clotilde Cerruti** — EPFL Neuro-X MSc
- **Jun Hao Zhou** — EPFL Neuro-X MSc

