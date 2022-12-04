library(SimDesign)

sparse.eigenvalue.simulator <- function(d, prop_sparse, mu0, mu1, mu2, sd){
  num.sparse <- floor(d*prop_sparse)
  first.eigenvalue <- abs(rnorm(1, mean = mu0, sd = sd))
  eigen.difference.set.1 <- abs(rnorm(n = d - num.sparse - 1, mean = mu1, sd = sd))
  eigen.difference.set.2 <- abs(rnorm(n = num.sparse, mean = mu2, sd = sd))
  eigen.differences <- c(eigen.difference.set.1, eigen.difference.set.2)
  eigenvalues <- rep(0,d)
  eigenvalues[1] <- exp(first.eigenvalue)
  for(i in 2:d){
    eigenvalues[i] <- exp(first.eigenvalue - sum(eigen.differences[1:(i-1)]))
  }
  return(eigenvalues)
}

# Example of generated eigenvalues.
sparse.eigenvalue.simulator(d = 25, prop_sparse = .6, mu0 = log(5), mu1 = .1, mu2 = 1, sd = .2)

shared.covariance.generator.eigen <- function(eigenvalues){
  Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = length(eigenvalues), R = length(eigenvalues))
  Generated.covariance <- Generated.orthogonal.matrix%*%diag(sqrt(eigenvalues))
  return(Generated.covariance)
}

set.seed(1234)
eigen(shared.covariance.generator.eigen(Generated.data$Eigenvalues))


noise.covariance.generator <- function(d, noise){
  covariance.noise <- matrix(0, nrow = d, ncol = d)
  for(i in 1:d){
    covariance.noise[i,i] <- noise
  }
  return(covariance.noise)
}


PCA.normal.data.generator <- function(n, d, prop_sparse, mu0, mu1, mu2, sd, data_mu, noise.pca){
  view.noise <- noise.covariance.generator(d = d, noise = noise.pca)
  eigenvalues <- sparse.eigenvalue.simulator(d = d, prop_sparse = prop_sparse, mu0 = mu0, mu1 =  mu1, mu2 = mu2, sd = sd)
  shared.matrix <- shared.covariance.generator.eigen(eigenvalues)
  generated.data <- shared.matrix%*%t(rmvnorm(n = n, mean = rep(data_mu, ncol(shared.matrix)), sigma = diag(noise.pca,ncol(shared.matrix))))
  CCA.list <- list(eigenvalues, shared.matrix, generated.data)
  names(CCA.list) = c("Eigenvalues", "Shared Matrix", "Generated Data")
  return(CCA.list)
}

library(rstan)
library(extraDistr)
n = 1000
d = 12
prop_anom = .4
desired_first_eigen_mean = 5

set.seed(11141)
Generated.data <- PCA.normal.data.generator(n = n, d = d, prop_sparse = prop_anom, mu0 = log(5), mu1 = .1, mu2 = 1, sd = .2, data_mu = .5, noise.pca = .05)
num.of.zero <- sum(Generated.data$Eigenvalues< .1)
num.of.anom <- floor(d*prop_anom)
num.of.non.anom <- d - num.of.anom
difference_of_logarithms <- -diff(log(Generated.data$Eigenvalues))
difference_of_logarithms
Generated.data$Eigenvalues



PCA.data <- list(
  N = n,
  D = d,
  Y = t(Generated.data$`Generated Data`),
  K = 2
)


simulated_eigenvalues <- data.frame(Generated.data$Eigenvalues)
fit.householder.desktop.1 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Eigenvalue mixture PPCA.stan", data = PCA.data, chains = 9, seed = 781, iter = 250, control = list(adapt_delta = .99))

fit.householder.desktop.new <- stan(file = "D:/School/Projects/RHouseholder/Eigenvalue mixture PPCA new.stan", data = PCA.data, chains = 4, seed = 781, iter = 250, control = list(adapt_delta = .99))


summary(fit.householder.desktop.1, pars = c("eigen_roots"))
summary(fit.householder.desktop.1, pars = c("mixture_proportions"))

dim.ratio <- function(CCA_data){
  return((CCA_data$Q)/CCA_data$N)
}

dim.ratio(CCA_data = CCA.data)

estimated.zeros <- function(fit){
  sum(summary(fit, pars = c("eigen_roots"))$summary[ , "50%"] < 1)
}

estimated.nonzeros <- function(fit){
  sum(summary(fit, pars = c("eigen_roots"))$summary[ , "50%"] >= 1)
}

posterior.mean.squared <- function(generated_data, fit){
  true.eigenvalues <- generated_data$`sparse shared eigenvalues`
  median.eigenvalues <- summary(fit, pars = c("eigen_roots"))$summary[ , "50%"]
  error <- rep(0,sum(Generated.data$`sparse shared eigenvalues` < 1))
  for(i in 1:length(error)){
    error[i] = true.eigenvalues[i] - median.eigenvalues[i]
  }
  return(mean(error^2))
}

true.zeros <- function(generated_data){
  return(sum(generated_data$`sparse shared eigenvalues` < 1))
}

true.nonzeros <- function(generated_data){
  return(sum(generated_data$`sparse shared eigenvalues` >= 1))
}

eigenvalue.proportion <- function(CCA_data){
  return(CCA_data$k/CCA_data$D)
}

eigenvalue.proportion(CCA_data = CCA.data)

true.zeros(generated_data = Generated.data)

true.nonzeros(generated_data = Generated.data)

estimated.zeros(fit = fit.householder.desktop.1)




