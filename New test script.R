require(rstiefel) # This will allow the easy generation of random rectangular orthogonal matrices via haar measure over the space of orthogonal matrices

require(mvtnorm) # This will be used for generating multivariate normal observations for our artificial datasets

eigenvalue.simulator <- function(dims, alpha, beta){
  eigenvalues.simulated <- sort(rgamma(dims, shape = alpha, rate = beta), decreasing = T)
  return(eigenvalues.simulated)
}

sparse.eigenvalue.simulator <- function(eigenvalues, proportion){
  number.to.shrink <- floor(length(eigenvalues)*proportion) - 1
  eigenvalues[(length(eigenvalues) - number.to.shrink):length(eigenvalues)] <- sort(runif( n = number.to.shrink + 1, min = 0, max = min(eigenvalues)), decreasing = T)
  return(eigenvalues)
}

shared.covariance.generator.eigen <- function(eigenvalues){
  Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = length(eigenvalues), R = length(eigenvalues))
  Generated.covariance <- Generated.orthogonal.matrix%*%diag(eigenvalues)
  return(Generated.covariance)
}

noise.covariance.generator <- function(data1.dim, data2.dim, shared.noise.1, shared.noise.2){
  covariance.noise <- matrix(0, nrow = data1.dim + data2.dim, ncol = data1.dim + data2.dim)
  noise.1 <- shared.noise.1
  noise.2 <- shared.noise.2
  for(i in 1:data1.dim){
    covariance.noise[i,i] <- noise.1
  }
  for(i in (data1.dim + 1):(data1.dim + data2.dim)){
    covariance.noise[i,i] <- noise.2
  }
  return(covariance.noise)
}

column.variance.generator <- function(view1.dim, view2.dim){
  column.variances <- runif(view1.dim + view2.dim, 0,1)
  return(column.variances)
}

view.specific.matrix.generator <- function(view1.dim, view2.dim, data1.dim, data2.dim,
                                           column.variances){
  require(mvtnorm)
  view.matrix <- matrix(0, nrow = data1.dim + data2.dim, ncol = view1.dim + view2.dim)
  for(j in 1:(view1.dim + view2.dim)){
    view.matrix[,j] <- as.vector(rmvnorm(1, mean = rep(0, data1.dim + data2.dim),
                                 sigma = diag(rep(column.variances[j], data1.dim + data2.dim), nrow = data1.dim + data2.dim)))
    for(i in 1:(data1.dim + data2.dim)){
      if(i > data1.dim && j <= view1.dim){
        view.matrix[i,j] = 0
      }
      else if(i <= data1.dim && j > view1.dim){
        view.matrix[i,j] = 0
      }
  }
  }
  return(view.matrix)
}

set.seed(123)

shared.eigenvalues <- eigenvalue.simulator(dims = 10, alpha = 2, beta = 1)
sparse.eigenvalues <- sparse.eigenvalue.simulator(eigenvalues = shared.eigenvalues, proportion = .5)
B <- shared.covariance.generator.eigen(sparse.eigenvalues)
view.noise <- noise.covariance.generator(data1.dim = 4, data2.dim = 6, shared.noise.1 = 1, shared.noise.2 = 1.1)
column.variances <- column.variance.generator(view1.dim = 2, view2.dim = 3)
view.matrix <- view.specific.matrix.generator(view1.dim = 2, view2.dim = 3, data1.dim = 4, data2.dim = 6, column.variances = column.variances)

Y <- B%*%t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)))) + 
  t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)) + view.noise))


data = list(N=n, D=d, K=k, ones=1, 
            y=Y)

set.seed(123)
library(rstan)
n = 300
D1 = 4
D2 = 6
K1 = 2
K2 = 3
k = K1 + K2
d = D1 + D2

init.init.list = list(sigma_weight = 4*sort(runif((d), min = 0, max = 1), decreasing = F),
                      sigma = sort(rbeta((d), shape1 = .1, shape2 = .1), decreasing = F))

init.list = list(init.init.list)
names(init.init.list) = c("sigma_weight", "sigma")
names(init.list) = c("chain 1")

Y.new <- view.matrix %*% t(rmvnorm(n = n, mean = rep(0, ncol(view.matrix)), sigma = diag(ncol(view.matrix)))) + 
  B%*%t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)))) + 
  t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)) + view.noise))

CCA.data <- list(
  N = n,
  Y = t(Y.new),
  D = d,
  D_1 = 4,
  D_2 = 6,
  K_1 = 2,
  K_2 = 3,
  Q = d
)

fit.householder.CCA <- stan(file = "D:/School/Projects/GitMCMCHouseholder/RHouseholder/PPCA_House_Test_ARD_Extended.stan",
                            data = CCA.data, chains = 1, seed = 1234,
                            control = list(max_treedepth = 12),
                            init = init.list, iter = 300)

summary(fit.householder.CCA, pars = c("sigma"))$summary
summary(fit.householder.CCA, pars = c("sigma_weight"))$summary
summary(fit.householder.CCA, pars = c("sigma_new"))$summary
summary(fit.householder.CCA, pars = c("view1_noise"))$summary
summary(fit.householder.CCA, pars = c("view2_noise"))$summary
summary(fit.householder.CCA, pars = c("view_matrix"))$summary





fit.householder <- stan(file = "D:/School/Projects/GitMCMCHouseholder/RHouseholder/PPCA_House_Test.stan", data = data.house, chains = 1, init = init.list, iter = 300)

summary(fit.householder, pars = c("sigma_new"))$summary

summary(fit.householder, pars = c("W"))$summary




