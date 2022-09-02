require(rstiefel) # This will allow the easy generation of random rectangular orthogonal matrices via haar measure over the space of orthogonal matrices

require(mvtnorm) # This will be used for generating multivariate normal observations for our artificial datasets

n = 500
d = 5
k = 5

eigenvalue.simulator <- function(dims, alpha, beta){
  eigenvalues.simulated <- sort(rgamma(dims, shape = alpha, rate = beta), decreasing = T)
  return(eigenvalues.simulated)
}

shared.covariance.generator.eigen <- function(eigenvalues){
  Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = length(eigenvalues), R = length(eigenvalues))
  Generated.covariance <- Generated.orthogonal.matrix%*%diag(eigenvalues)
  return(Generated.covariance)
}

shared.eigenvalues <- eigenvalue.simulator(dims = 5, alpha = 1, beta = 1)
B <- shared.covariance.generator.eigen(shared.eigenvalues)

Y <- B%*%t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)))) + 
  t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B))))



data = list(N=n, D=d, K=k, ones=1, 
            y=Y)

library(rstan)

file.ARD.test<- "C:/Users/qsimo/Documents/Code/RHouseholder/New test code.stan"

fit.ARD.test <- stan(file = file.ARD.test, data = data)


summary(fit.ARD.test, pars = c("sigma"))$summary

summary(fit.ARD.test, pars = c("A"))$summary


PPCA.horseshoe = "
data {
  int<lower=1> N;             // num datapoints
  int<lower=1> D;              // num dimension
  int<lower=1> K;              // num basis
  real<lower=0> ones;
  real y[D,N];
}
transformed data {
  matrix[K,K] Sigma;
  vector<lower=0>[K] diag_elem;
  vector<lower=0>[K] zr_vec;
  for (k in 1:K) zr_vec[k] <- 0;
  for (k in 1:K) diag_elem[k] <- ones;
  Sigma <- diag_matrix(diag_elem);
}
parameters {
  matrix[D,K] A;             // basis
  vector[K] x[N];            // coefficients
  real<lower=0> sigma;       // noise variance
  vector<lower=0>[K] lambda; // the local prior on the weights
  real<lower=0> tau;         // the global prior on the weights        
}
model {  
  
  lambda ~ cauchy(0,1);
  tau ~ cauchy(0,1);
  
  for (i in 1:N)
      x[i] ~ multi_normal(zr_vec, Sigma);
      
  for (d in 1:D)
    for (k in 1:K)
      A[d,k] ~ normal(0, lambda[k]*lambda[k]*tau*tau);
  
  for (i in 1:N)
    for (d in 1:D)
      //y[d,i] ~ normal(dot_product(row(A, d), x[i]), sigma);
      increment_log_prob(normal_log(y[d,i], dot_product(row(A, d), x[i]), sigma));
}
generated quantities {
    vector[N] log_lik;
    for (n in 1:N)
       for (d in 1:D)
        log_lik[n] <- normal_log(y[d,n], dot_product(row(A, d), x[n]), sigma);
}


"

fit.horseshoe <- stan(model_code = PPCA.horseshoe, data = data)


summary(fit.horseshoe, pars = c("sigma"))$summary

summary(fit.horseshoe, pars = c("A"))$summary

data.house <- list(
  N = n,
  Y = t(Y),
  D = d,
  Q = k
)
fit.householder <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/PPCA_House.stan", data = data.house)

summary(fit.householder, pars = c("sigma"))$summary

summary(fit.householder, pars = c("W"))$summary
