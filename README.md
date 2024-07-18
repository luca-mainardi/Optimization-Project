# Optimization for Data Science - Political Districting Project

## Overview
This repository contains the project developed for the course "Optimization for Data Science" (2AMS50) at Eindhoven University of Technology. The project focuses on solving the problem of political districting, which aims to divide geographical regions into electoral districts that ensure fair representation and equitable distribution of political power among voters.

## Project Contributors
- **Luca Mainardi** (2014602)
- **Laurynas Jagutis** (2037637)
- **Ian van de Wetering** (1009805)
- **Błażej Nowak** (1617303)
- **Thomas Warnier** (1423495)

## Project Description
The project investigates methods to create optimal political districts, adhering to the constraints of equal population distribution, contiguity, and compactness. The project utilizes integer programming to find exact solutions and metaheuristics, specifically simulated annealing, to provide heuristic solutions. The solutions are analyzed using data from 50 US states, divided into counties.


## Getting Started
### Prerequisites
- Python 3.x
- Gurobi Optimization Software
- Necessary Python libraries: `gurobipy`, `gerrychain`, `geopandas`, `networkx`


### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/luca-mainardi/Optimization-Project.git
    cd Optimization-Project
    ```

2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure Gurobi is installed and properly configured.

