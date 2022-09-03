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
  matrix[D,K] A;            // basis
  vector[K] x[N];           // coefficients
  real<lower=0> sigma;      // noise variance
  vector<lower=0>[K] alpha; // prior on weights
}
model {  
  
  for (i in 1:N)
      x[i] ~ multi_normal(zr_vec, Sigma);
      
  for (d in 1:D)
    for (k in 1:K)
      A[d,k] ~ normal(0, alpha[k]);
  
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