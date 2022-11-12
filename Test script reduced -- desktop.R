wishart.matrix.generator <- function(n.row, n.col){
  generated.matrix <- matrix(0, nrow = n.row, ncol = n.col)
  for(i in 1:nrow(generated.matrix)){
    for(j in 1:ncol(generated.matrix)){
      generated.matrix[i,j] = rnorm(1)
    }
  }
  return(generated.matrix %*% t(generated.matrix))
}

sparse.eigenvalue.simulator <- function(eigenvalues, proportion, max, shrinkage){
  number.to.shrink <- floor(length(eigenvalues)*proportion) - 1
  eigenvalues[(length(eigenvalues) - number.to.shrink):length(eigenvalues)] <- shrinkage*sort(runif( n = number.to.shrink + 1, min = 0, max = max), decreasing = T)
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
  column.variances <- extraDistr::rhcauchy(view1.dim + view2.dim)
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

CCA.normal.data.generator <- function(n, D1, D2, K1, K2, prop.CCA){
  k = K1 + K2
  d = D1 + D2
  view.specific.column.variances <- column.variance.generator(K1, K2)
  view.specific.matrix <- view.specific.matrix.generator(view1.dim = K1, view2.dim = K2,
                                                         data1.dim = D1, data2.dim = D2,
                                                         column.variances = view.specific.column.variances)
  wishart.matrix <- wishart.matrix.generator(n.row = d, n.col = d)
  wishart.eigenvalues <- sqrt(eigen(wishart.matrix)$values)
  sparse.wishart.eigenvalues <- sparse.eigenvalue.simulator(wishart.eigenvalues, proportion = prop.CCA, max = .1, shrinkage = .9)
  shared.matrix <- shared.covariance.generator.eigen(sparse.wishart.eigenvalues)
  view.noise <- noise.covariance.generator(data1.dim = D1, data2.dim = D2, shared.noise.1 = .3, shared.noise.2 = .5)
  generated.data <- view.specific.matrix %*% t(rmvnorm(n = n, mean = rep(0, ncol(view.specific.matrix)), sigma = diag(ncol(view.specific.matrix)))) + 
    shared.matrix%*%t(rmvnorm(n = n, mean = rep(0, ncol(shared.matrix)), sigma = diag(ncol(shared.matrix)))) + 
    t(rmvnorm(n = n, mean = rep(0, ncol(shared.matrix)), sigma = diag(ncol(shared.matrix)) + view.noise))
  
  CCA.list <- list(view.specific.column.variances, view.specific.matrix, wishart.eigenvalues,
                   sparse.wishart.eigenvalues, shared.matrix, generated.data)
  names(CCA.list) = c("view specific column variances", "view matrix", "original shared eigenvalues",
                      "sparse shared eigenvalues", "shared matrix", "generated data")
  return(CCA.list)
}

library(rstan)
library(extraDistr)
n = 1000
D1 = 12
D2 = 6
K1 = 2
K2 = 1
k = K1 + K2
d = D1 + D2

Generated.data <- CCA.normal.data.generator(n = n, D1 = D1, D2 = D2, K1 = K1, K2 = K2, prop.CCA = .8)

eigen.variance <- var(-diff(log(Generated.data$`original shared eigenvalues`)))


num.of.zero <- sum(Generated.data$`sparse shared eigenvalues`< .1)

CCA.data <- list(
  N = n,
  Y = t(Generated.data$'generated data'),
  D = d,
  D_1 = D1,
  D_2 = D2,
  K_1 = K1,
  K_2 = K2,
  Q = d,
  k = k,
  eigenvalue_tuning_number = 1
)

set.seed(4333213)
uni.max = max(Generated.data$`sparse shared eigenvalues`) + runif(1, min = 0, max = max(Generated.data$`sparse shared eigenvalues`)/10)
chain.1 = list(eigen_weight = sort(rbeta((d), shape1 = .1, shape2 = .1), decreasing = T),
               eigen_roots = sort(c(runif(d/2 , min = 0, max = 1),
                                    uni.max*runif(d/2 , min = .25 , max = 1)), decreasing = T),
               column.variances = sort(rhcauchy(K1 + K2), decreasing = F))

chain.1 = list(column.variances = sort(rhcauchy(K1 + K2), decreasing = F))

names(chain.1) = c("column_variances")

simulated_eigenvalues <- data.frame(Generated.data$`sparse shared eigenvalues`)

fit.householder.desktop.1 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Eigenvalue mixture.stan", data = CCA.data, chains = 9, seed = 3021, iter = 250, control = list(adapt_delta = .99))


summary(fit.householder.desktop.1, pars = c("eigen_roots"))
summary(fit.householder.desktop.1, pars = c("eigen_differences"))
summary(fit.householder.desktop.1, pars = c("eigen_max"))
summary(fit.householder.desktop.1, pars = c("mixture_proportions"))
summary(fit.householder.desktop.1, pars = c("eigen_variance"))




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

estimated.zeros(fit = fit.householder.desktop.3)

posterior.mean.squared(generated_data = Generated.data,
                       fit = fit.householder.desktop.3)








stan_trace(fit.householder.desktop.1, pars = c("eigen_roots[1]","eigen_roots[2]", 
                                               "eigen_roots[3]", "eigen_roots[4]"))

stan_trace(fit.householder.desktop.2, pars = c("eigen_roots[13]","eigen_roots[14]", 
                                               "eigen_roots[15]", "eigen_roots[16]"))

stan_trace(fit.householder.desktop.3, pars = c("eigen_roots[1]","eigen_roots[2]", 
                                               "eigen_roots[3]", "eigen_roots[4]"))





plot(summary(fit.householder.desktop, pars = c("eigen_roots")))