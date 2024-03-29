# This code provides artificial data for doing inference in the sparse CCA problem

require(rstiefel) # This will allow the easy generation of random rectangular orthogonal matrices via haar measure over the space of orthogonal matrices

require(mvtnorm) # This will be used for generating multivariate normal observations for our artificial datasets

# For the shared variation, I will generate the loading matrix via it's eigenvalue/svd decomposition
# by generating eigenvalues from the order statistics of a distribution with high mass near 0. This can
# be fulfilled from a gamma(1,2), gamma(2,2), or something similar.

# Generating the eigenvalues:

eigenvalue.simulator <- function(dims, alpha, beta){
  eigenvalues.simulated <- vector(mode = "list", length = length(dims))
  for(i in 1:length(dims)){
    eigenvalues.simulated[[i]] <- -sort(-rgamma(dims[i], shape = alpha, rate = beta))
  }
  return(eigenvalues.simulated)
}

# Note that eigenvalue.simulator will not necessarily produce a strong eigengap, instead, we will use the following function to create a produce a larger eigengap in our matrix.

# This function will return eigenvalues in the same way as eigenvalue.simulator, but will additionally
# push eigenvalues to a small value with probability "percent".

# What exactly will this small value be? It will be some number between 0 and the smallest eigenvalue
# generated. This will be done by multiplying our vector of eigenvalues with a vector generated by the
# following function:

# There are two inputs to this function:

# eigenvalue.list - a list of one of more vectors of eigenvalues sorted highest to lowest.
# percent - percent of eigenvalues to be shrunk towards 0.

eigenvalue.shrinker <- function(eigenvalue.list, percent){
  require(purrr)
  eigenvalue.magnification <- vector(mode = "list", length = length(eigenvalue.list))
  for(i in 1:length(eigenvalue.list)){
    eigen.min <- min(eigenvalue.list[[i]])
    for(j in 1:length(eigenvalue.list[[i]])){
      a = purrr::rbernoulli(1, p = percent)
      eigenvalue.list[[i]][j] <- eigenvalue.list[[i]][j]*runif(1, min = 0, max = eigen.min)*a + eigenvalue.list[[i]][j]*(1 - a)
    }
  }
  return(eigenvalue.list)
}

# I think it would make more sense if eigenvalue.shrinker pushed only the smaller eigenvalues towards 0 to induce a more natural eigengap.

eigenvalue.simulator.sparse <- function(dims, alpha, beta, percent){
  require(purrr)
  eigenvalues <- eigenvalue.simulator(dims = dims, alpha = alpha, beta = beta)
  eigenvalues.shrunk <- eigenvalue.shrinker(eigenvalues, percent = percent)
  for(i in 1:length(eigenvalues)){
    eigenvalues[[i]] <- -sort(-eigenvalues.shrunk[[i]])
  }
  return(eigenvalues)
}


# This will generate the final representation of the shared loading matrix for Bayesian CCA

shared.covariance.generator <- function(data.dim, shared.dim, alpha, beta, percent){
  matrix.list <- vector(mode = "list", length = length(shared.dim))
  Generated.eigenvalues <- eigenvalue.simulator.sparse(dims = shared.dim, alpha = alpha, beta = beta,
                                                       percent = percent)
  print(c("The eigenvalues of the shared covariance are:", Generated.eigenvalues))
  for(i in 1:length(shared.dim)){
    Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = data.dim, R = shared.dim[i])
    matrix.list[[i]] <- Generated.orthogonal.matrix%*%diag(Generated.eigenvalues[[i]])%*%t(Generated.orthogonal.matrix)
  }
  return(matrix.list)
}



# We also need a way to generate the loading matrix of our data. I will propose generating our
# loading matrix with normal entries (with different covariance for each column), where the views
# will be generated with a bernoulli.


view.specific.matrix.generator <- function(view1.dim, view2.dim, data1.dim, data2.dim,
                                           variance.params){
  generated.matrix <- matrix(0, ncol = (view1.dim + view2.dim), nrow = (data1.dim + data2.dim))
  for(j in 1:(view1.dim + view2.dim)){
    if(j <= view1.dim){
      column.variance.1 <- rgamma(1, shape = variance.params[1], rate = variance.params[2])
      for(i in 1:data1.dim){
        generated.matrix[i,j] <- rnorm(1, mean = 0, sd = column.variance.1)
      }
    }
      else if(j >= view1.dim){
        column.variance.2 <- rgamma(1, shape = variance.params[1], rate = variance.params[2])
        for(i in (data1.dim + 1):(data1.dim + data2.dim)){
          generated.matrix[i,j] <- rnorm(1, mean = 0, sd = column.variance.2)
        }
      }
      for(i in 1:(data1.dim + data2.dim)){
        if(i > data1.dim && j <= view1.dim){
          generated.matrix[i,j] = 0
        }
        else if(i <= data1.dim && j > view1.dim){
          generated.matrix[i,j] = 0
        }
      }
  }
  print(c("The view-specific loading matrix is", generated.matrix))

  return(generated.matrix)
}

noise.covariance.generator <- function(data1.dim, data2.dim, alpha, beta){
  covariance.noise <- matrix(0, nrow = data1.dim + data2.dim, ncol = data1.dim + data2.dim)
  noise.1 <- rgamma(1, shape = alpha, rate = beta)
  noise.2 <- rgamma(1, shape = alpha, rate = beta)
  for(i in 1:data1.dim){
    covariance.noise[i,i] <- noise.1
  }
  for(i in (data1.dim + 1):(data1.dim + data2.dim)){
    covariance.noise[i,i] <- noise.2
  }
  return(covariance.noise)
}

# Investigation of appropriate distributional choices for the noise variance

x <- seq(0,1, by = .01)

plot(x, dbeta(x, shape1 = .5, shape2 = 2), type = "l", col = "red")





# Data Generation

K.true = c(4,15,30) # True dimensionality of the shared latent variable (number of non-zero eigenvalues)

Kv.1.true = c(2,4,10,15,40) # True dimensionality of the latent variable for view-specific variation (will be interpreted )

Kv.2.true = c(6,10,13,21,52)

# As an example line of code for generating the shared covariance:

shared.covariance.generator(data.dim = 1000, shared.dim = K.true, alpha = 1, beta = 1, percent = .7)


# And an example line of code for generating the view-specific loading matrix:

view.specific.matrix.generator(view1.dim = 7, view2.dim = 10, data1.dim = 12, data2.dim = 15, variance.params = c(1,1))


CCA.dataset.generator <- function(N.observations, data1.dim, data2.dim, view1.dim, view2.dim, shared.dim, alpha.noise, beta.noise, alpha.eigenvalue, beta.eigenvalue,
                                  column.loading.variance.parameters, percent){
  Y <- matrix(0, nrow = data1.dim + data2.dim, ncol = N.observations)
  noise.covariance <- noise.covariance.generator(data1.dim = data1.dim, data2.dim = data2.dim, alpha = alpha.noise, beta = beta.noise)
  shared.covariance <- shared.covariance.generator(data.dim = data1.dim + data2.dim, shared.dim = shared.dim, alpha = alpha.eigenvalue, beta = beta.eigenvalue, percent = percent)[[1]]
  view.loading.matrix <- view.specific.matrix.generator(view1.dim = view1.dim, view2.dim = view2.dim, data1.dim = data1.dim, data2.dim = data2.dim,
                                                        variance.params = column.loading.variance.parameters)
  for(j in 1:N.observations){
    z <- rmvnorm(1, mean = rep(0, view1.dim + view2.dim), sigma = diag(view1.dim + view2.dim))
    Y[,j] <- rmvnorm(1, mean = view.loading.matrix%*%t(z), sigma = shared.covariance + noise.covariance)
  }
  return(Y)
}


# An example line for generating the CCA dataset:

CCA.dataset.generator(N.observations = 100, data1.dim = 10, data2.dim = 13, view1.dim = 3, view2.dim = 6, shared.dim = 23, alpha.noise = 1,
                      beta.noise = 1, alpha.eigenvalue = 1, beta.eigenvalue = 1, column.loading.variance.parameters = c(1,1), percent = .7)





# Here I will begin the task of fitting the model from "Sparse Householder CCA" with Rstan:

library(rstan)

# This will be our first "true" data:


set.seed(1234)
Y <- t(CCA.dataset.generator(N.observations = 300, data1.dim = 3, data2.dim = 4, view1.dim = 1, view2.dim = 2, shared.dim = 7, alpha.noise = 1,
                           beta.noise = 1, alpha.eigenvalue = 3, beta.eigenvalue = 2, column.loading.variance.parameters = c(1,1), percent = .7))

simulation.data <- list(
  N = nrow(Y),
  D_1 = 3,
  D_2 = 4,
  K_1 = 1,
  K_2 = 2,
  D = ncol(Y),
  Q = 7,
  Y = Y[[5]]
)


file.CCA <- "C:/Users/qsimo/Documents/Code/RHouseholder/CCA_House_Troubleshooting.stan"



fit.CCA <- stan_model(file.CCA)


fit <- sampling(fit.CCA, data = simulation.data)

fit.model <- stan(file = file.CCA, data = simulation.data, chains = 1, iter = 500, warmup = 300)




file.CCA.laptop <- "D:/School/Projects/GitMCMCHouseholder/RHouseholder/CCA_House_Troubleshooting.stan"

fit.CCA.laptop <- stan_model(file.CCA.laptop)

samples.CCA <- sampling(fit.CCA.laptop, data = simulation.data, iter = 100)

print(samples.CCA)




plot(x,dbeta(x, shape1 = 1.13, shape2 = 1))



