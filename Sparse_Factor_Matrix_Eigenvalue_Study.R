#The purpose of this code is to do some EDA for the distribution of eigenvalues for the loading matrix in CCA. This assumption implies the observation, y, 
# has block dependency with a latent variable z.

N = 400
D = 30
p = 10 #Number of components that contribute to both Y_{a} and Y_{b}
n = 200 #Number of rows for first (and second) component
k1 = 10 #Number of components that contribute to Y_{a}.
k2 = D - k1 - p #Number of components that contribute to Y_{b} 
Loading.matrix <- matrix(,nrow = N, ncol = D)

for(i in 1:D){
  Loading.matrix[,i] <- rnorm(400)
}

for(k in (p + k1 + 1): D){
  for(j in 1:n){
    Loading.matrix[j,k] <- 0
  }
}


for(j in (p+1):(p + k1)){
  for(k in (n+1):N){
    Loading.matrix[k,j] <- 0
  }
}


head(Loading.matrix)

tail(Loading.matrix)


Z <- rnorm(D)
epsilon <- rnorm(D)
Y <- Loading.matrix %*% Z + rnorm(N)

ev.loading <- eigen(t(Loading.matrix) %*% Loading.matrix)

values <- ev.loading$values
values
hist(values)

vectors <- ev.loading$vectors

principal.value.matrix <- diag(values)

principal.value.matrix %*% vectors; Loading.matrix

tail(principal.value.matrix %*% vectors)


# Run five simulations and save the corresponding eigenvalues from each run:

Eigen.sim.matrix <- matrix(,nrow = 5, ncol = 30)


for(l in 1:5){
  
  for(i in 1:D){
    Loading.matrix[,i] <- rnorm(400)
  }
  
  for(k in (p + k1 + 1): D){
    for(j in 1:n){
      Loading.matrix[j,k] <- 0
    }
  }
  
  
  for(j in (p+1):(p + k1)){
    for(k in (n+1):N){
      Loading.matrix[k,j] <- 0
    }
  }
  
  Z <- rnorm(D)
  epsilon <- rnorm(D)
  Y <- Loading.matrix %*% Z + rnorm(N)
  
  ev.loading <- eigen(t(Loading.matrix) %*% Loading.matrix)
  values <- ev.loading$values
  Eigen.sim.matrix[l,] <- ev.loading$values
  
}


for(i in 1:5){
  hist(Eigen.sim.matrix[i,])
}

# For comparison, let's see what happens when we have a loading matrix completely composed of standard normals.

Eigen.sim.standard.normal.matrix <- matrix(,nrow = 5, ncol = 30)

standard.normal.matrix <- matrix(,nrow = N, ncol = D)

for(l in 1:5){
  
  for(i in 1:D){
    standard.normal.matrix[,i] <- rnorm(400)
  }
  
  Z <- rnorm(D)
  epsilon <- rnorm(D)
  Y <- standard.normal.matrix %*% Z + rnorm(N)
  
  ev.standard.loading <- eigen(t(standard.normal.matrix) %*% standard.normal.matrix)
  values <- ev.standard.loading$values
  Eigen.sim.standard.normal.matrix[l,] <- ev.standard.loading$values
}


for(i in 1:5){
  hist(Eigen.sim.standard.normal.matrix[i,])
}

par(mfrow = c(5,2))

for(i in 1:5){
  hist(Eigen.sim.matrix[i,])
}

for(i in 1:5){
  hist(Eigen.sim.standard.normal.matrix[i,])
}


#Now I want a function that generates t side by side plots of the distribution
# of the eigenvalues for the CCA loading matrix vs the standard normal matrix.

#Note the assumption of CCA here will be that there are two groups


# The inputs for the function are:

# N - Sample size
# D - Number of variables
# p - Number of non-zero interactions shared by Z and Y_{a} and Y_{b}
# k - Number of non-zero interactions shared by z and Y_{a} only.
# n - Number of observations for Y_{a}


eigenvalue.comparison <- function(N,D,p,k,n,t){
  par(mfrow  = c(t,2))
  Eigen.sim.standard.normal.matrix <- matrix(,nrow = t, ncol = D)
  Eigen.sim.non.standard.matrix <- matrix(, nrow = t, ncol = D)
  
  standard.normal.matrix <- matrix(,nrow = N, ncol = D)
  
  for(l in 1:t){
    
    for(i in 1:D){
      standard.normal.matrix[,i] <- rnorm(N)
    }
    
    Z <- rnorm(D)
    epsilon <- rnorm(D)
    Y <- standard.normal.matrix %*% Z + rnorm(N)
    
    ev.standard.loading <- eigen(t(standard.normal.matrix) %*% standard.normal.matrix)
    values <- ev.standard.loading$values
    Eigen.sim.standard.normal.matrix[l,] <- values
  
  
  
  Loading.matrix.non.standard <- matrix(,nrow = N, ncol = D)
  
  for(i in 1:D){
    Loading.matrix.non.standard[,i] <- rnorm(N)
  }
  
  for(v in (p + k + 1): D){
    for(j in 1:n){
      Loading.matrix.non.standard[j,v] <- 0
    }
  }
  
  
  for(j in (p+1):(p + k)){
    for(r in (n+1):N){
      Loading.matrix.non.standard[r,j] <- 0
    }
  }
  
  Z <- rnorm(D)
  epsilon <- rnorm(D)
  Y <- Loading.matrix.non.standard %*% Z + rnorm(N)
  
  ev.loading <- eigen(t(Loading.matrix.non.standard) %*% Loading.matrix.non.standard)
  values <- ev.loading$values
  Eigen.sim.non.standard.matrix[l,] <- ev.loading$values
  
  
  hist(Eigen.sim.standard.normal.matrix[l,])
  hist(Eigen.sim.non.standard.matrix[l,])
  }
}

eigenvalue.comparison(N = 500, D = 30, p = 10, k = 10, n = 200, t = 5)







