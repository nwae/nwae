
rm(list=ls())
# Avoid scientific notation such as 1e+05, so that when in Excel it will be exported correctly to DB
options(scipen=999)
setwd("~/git/nwae/nwae/analysis")

# How many dimensional space
DIM = 500
# How many simulations
N_SIMS = 100000

#
# Pick positive points at random
#
create.normalized.random.matrix <- function(
  dim,
  n.sims
)
{
  x = runif(DIM*N_SIMS)
  m = matrix(x, nrow = N_SIMS, ncol = DIM)
  mag = rowSums(m*m)**0.5
  # Normalize the rows
  m.norm = m/mag
  return(m.norm)
}

m1 = create.normalized.random.matrix(dim = DIM, n.sims = N_SIMS)
m2 = create.normalized.random.matrix(dim = DIM, n.sims = N_SIMS)
# Check if normalized
rowSums(m1*m1)
rowSums(m2*m2)

# Now calculate distances between them
dif = (m1-m2)
dist = rowSums(dif*dif)**0.5
mean(dist)

quantile(dist, c(1:20)/20)
