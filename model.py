import gurobipy as gp
from gurobipy import GRB
import csv

# _____________________________________ Constants _____________________________________

STATE = "TX"
UPPER_BOUND_PCT = 0.65

# _____________________________________ Load the data _____________________________________

# Distances between nodes matrix
distances = []
with open(f"data/{STATE}/{STATE}_distances.csv", "r") as file:
    reader = csv.reader(file)
    next(reader, None)  # Skip the header
    distances = [list(map(int, row[1:])) for row in reader]  # Read line skipping NodeID
print(f"Distances:\n{distances}")

# Number of nodes
nodes_count = len(distances)
print(f"Nodes count: {nodes_count}")

# Number of districts
districts_count = 0

# Adjacency matrix
adjacency_matrix = [[0 for _ in range(nodes_count)] for _ in range(nodes_count)]
with open(f"data/{STATE}/{STATE}.dimacs", "r") as file:
    for line in file:
        if line.startswith("e"):
            _, node1, node2 = line.split()
            node1 = int(node1)
            node2 = int(node2)
            adjacency_matrix[node1][node2] = 1
            adjacency_matrix[node2][node1] = 1
        if line.startswith("c"):
            _, _, d_count = line.split()
            districts_count = int(d_count)

print(f"Adjacency matrix:\n{adjacency_matrix}")
print(f"Districts count: {districts_count}")

# Population vector
population = []
total_population = 0
with open(f"data/{STATE}/{STATE}.population", "r") as file:
    total_population = int(file.readline().split()[-1])
    for line in file:
        population.append(int(line.split()[1]))

print(f"Population vector: {population}")
print(f"Total population: {total_population}")


# _____________________________________ Model _____________________________________

# Create a new model
model = gp.Model("example_model")

# Rest of the code...
m = nodes_count  # Number of territories
n = districts_count  # Number of districts
d = distances  # Distance matrix (m x m)
p = population  # Population vector of length m
A = adjacency_matrix  # Adjacency matrix (m x m)
a = 0  # Lower bound constraint on population difference
b = (
    total_population * UPPER_BOUND_PCT
)  # Upper bound constraint on population difference

# Decision variables
X = {}
S = {}
P = {}
R = {}
alpha = {}
beta = {}
gamma = {}
delta = {}

for i in range(m):
    for j in range(n):
        X[i, j] = model.addVar(vtype=GRB.BINARY, name=f"X_{i}_{j}")
        S[i, j] = model.addVar(vtype=GRB.BINARY, name=f"S_{i}_{j}")
        gamma[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"gamma_{i}_{j}")
        alpha[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"alpha_{i}_{j}")

for i in range(m):
    for k in range(m):
        for j in range(n):
            P[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"P_{i}_{k}_{j}")
            R[i, k, j] = model.addVar(vtype=GRB.BINARY, name=f"R_{i}_{k}_{j}")
            beta[i, k, j] = model.addVar(vtype=GRB.INTEGER, name=f"beta_{i}_{k}_{j}")

for j in range(n):
    delta[j] = model.addVar(vtype=GRB.INTEGER, name=f"delta_{j}")

# Objective function
obj = gp.quicksum(
    d[i][k] ** 2 * p[k] * R[i, k, j]
    for i in range(m)
    for k in range(m)
    for j in range(n)
)
model.setObjective(obj, GRB.MINIMIZE)  # Minimize objective function

# Constraints

# S[i, j] + X[k, j] - 1 <= 2 * R[i, k, j] <= S[i, j] + X[k, j]
for i in range(m):
    for k in range(m):
        for j in range(n):
            model.addConstr(S[i, j] + X[k, j] - 1 <= 2 * R[i, k, j])
            model.addConstr(2 * R[i, k, j] <= S[i, j] + X[k, j])

# Sum of X[i, j] for each row i equals 1
for i in range(m):
    model.addConstr(gp.quicksum(X[i, j] for j in range(n)) == 1)

# Sum of S[i, j] for each column j equals 1
for j in range(n):
    model.addConstr(gp.quicksum(S[i, j] for i in range(m)) == 1)

# S[i, j] <= X[i, j]
for i in range(m):
    for j in range(n):
        model.addConstr(S[i, j] <= X[i, j])

# 0 <= alpha[i, j] <= m*R[i, i, j]
for i in range(m):
    for j in range(n):
        model.addConstr(0 <= alpha[i, j])
        model.addConstr(alpha[i, j] <= m * R[i, i, j])

# X[i, j] + S[i, j] - 1 <= 2*R[i, i, j] <= X[i, j] + S[i, j]
for i in range(m):
    for j in range(n):
        model.addConstr(X[i, j] + S[i, j] - 1 <= 2 * R[i, i, j])
        model.addConstr(2 * R[i, i, j] <= X[i, j] + S[i, j])

# Sum of alpha[i, j] equals m
model.addConstr(gp.quicksum(alpha[i, j] for i in range(m) for j in range(n)) == m)

# 0 <= beta[i, k, j] <= m*P[i, k, j]*A[i, k]
for i in range(m):
    for j in range(n):
        for k in range(m):
            model.addConstr(0 <= beta[i, k, j])
            model.addConstr(beta[i, k, j] <= m * P[i, k, j] * A[i][k])

# X[i, j] + X[k, j] - 1 <= 2*P[i, k, j] <= X[i, j] + X[k, j]
for i in range(m):
    for j in range(n):
        for k in range(m):
            model.addConstr(X[i, j] + X[k, j] - 1 <= 2 * P[i, k, j])
            model.addConstr(2 * P[i, k, j] <= X[i, j] + X[k, j])

# 0 <= gamma[i, j] <= X[i, j]
for i in range(m):
    for j in range(n):
        model.addConstr(0 <= gamma[i, j])
        model.addConstr(gamma[i, j] <= X[i, j])

# Sum of gamma[i, j] equals delta[j]
for j in range(n):
    model.addConstr(gp.quicksum(gamma[i, j] for i in range(m)) == delta[j])

# Additional constraints
for l in range(m):
    for j in range(n):
        model.addConstr(
            alpha[l, j] + gp.quicksum(beta[i, l, j] * A[i][l] for i in range(m))
            == gamma[l, j] + gp.quicksum(beta[l, k, j] * A[l][k] for k in range(m))
        )

# Bounds on difference of population between districts
for j in range(n):
    model.addConstr(a <= gp.quicksum(p[i] * X[i, j] for i in range(m)))
    model.addConstr(gp.quicksum(p[i] * X[i, j] for i in range(m)) <= b)

# Optimize model
model.optimize()

# Print optimal solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")
    # Print values of decision variables
    for v in model.getVars():
        if v.VarName.startswith("X"):
            print(f"{v.varName} = {v.x}")
    # Print objective value
    print(f"Objective value = {model.objVal}")
else:
    print("Optimal solution not found.")
