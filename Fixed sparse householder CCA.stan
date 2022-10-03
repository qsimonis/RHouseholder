parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    vector<lower = 0>[Q] eigen_differences;
    vector<lower = 0, upper = 1>[Q] local_variance;
    real<lower = 0> tau_1;
    real<lower = 0> tau_2;
    real<lower = 0> eigen_variance;
    matrix[D , K_1 + K_2] B;
    vector<lower = 0, upper = 1>[K_1 + K_2] beta_column;
    vector<lower = 0>[K_1 + K_2] alpha;
    vector[K_1 + K_2] X[N];
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    matrix[D, K_1 + K_2] view_matrix;
    vector[D] mu[N];
    positive_ordered[Q] weight_variance;
    vector[Q] eigen_roots;
    vector[Q] eigen_max;
    vector[Q] relative_min;
    {
      for(i in 1:Q){
        eigen_roots[i] = eigen_roots[(Q - i) + 1];
      }
    }
    {
      eigen_differences[1] = 2*log(eigen_roots[1]);
      print("1:", "eigen_differences[1]", eigen_differences[1]);
      print("q", 1, "eigen_roots[q]", eigen_roots[1]);
      for(q in 2:Q){
        eigen_differences[q] = 2*log(eigen_roots[q-1]+.01) - 2*log(eigen_roots[q] + .01);
        print("q", q, "eigen_roots[q]", eigen_roots[q]);
        print("q:", q,  "eigen_differences[q]", eigen_differences[q]);
      }
    }
    {
      eigen_max[1] = eigen_differences[1];
      for(q in 2:Q){
        eigen_max[q] = max(eigen_differences[2:q]);
      }
    }
    {
      relative_min[1] = eigen_differences[1];
      for(q in 2:Q){
        relative_min[q] = min(eigen_differences[1:q]);
      }
    }
    
    {
      for(j in 1:(K_1 + K_2)){
        for(i in 1:D){
          if((j > K_1 && i <= D_1) || (j <= K_1 && i > D_1)){
            view_matrix[i,j] = 0;
          }
          else if (j <= K_1 && i <= D_1){
            view_matrix[i,j] = view_vector[j*i];
          }
          else if (j > K_1 && i > D_1){
            view_matrix[i,j] = view_vector[(j-K_1)*(i-D_1) + K_1*D_1];
          }
        }
      }
    }
    
    for(i in 1:N){
      mu[i] = view_matrix*X[i];
    }

    {
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;

        W = U*diag_matrix(eigen_roots);

        K = W*W';
        for (d in 1:D_1)
            K[d, d] = K[d,d] + square(tau_1) + 1e-14;
        for (d in 1:D_2){
          K[d,d] = K[d,d] + square(tau_2) + 1e-14;
        }
        L = cholesky_decompose(K);
    }
}
model{
    tau_1 ~ cauchy(0,1);
    tau_2 ~ cauchy(0,1);
    beta_column ~ beta(.001,.001);
    alpha ~ cauchy(0,1);
    eigen_variance ~ cauchy(0,1);
    
    for(q in 1:Q){
      local_variance[q] ~ beta(min(eigen_differences[1:q]), max(eigen_differences[1:q]));
    }

    for(j in 1:(K_1 + K_2)){
      column_variances[j] ~ cauchy(0,1);
    }
    
    for(i in 1:D){
      for(j in 1:(K_1 + K_2)){
        if(i <= D_1 && j <= K_1){
          view_vector[i*j] ~ normal(0,column_variances[j]);
        }
        else if(i > D_1 && j > K_1){
          view_vector[(i-D_1)*(j - K_1) + D_1*K_1] ~ normal(0, column_variances[j]);
        }
      }
    }

    v ~ normal(0,1);

    //prior on singular values
    eigen_roots[1] ~ exponential(10);
    for(q in 2:Q){
      eigen_roots[q] ~ normal(0, eigen_variance*local_variance[q - 1]);
    }
    target += sum(2*log(eigen_roots));
    
    for(i in 1:N){
      X[i] ~ multi_normal(rep_vector(0, K_1 + K_2),diag_matrix(rep_vector(1,K_1 + K_2)));
      Y[i] ~ multi_normal_cholesky(B*X[i], L);
    }
}
generated quantities {
    matrix[D, Q] U_n = orthogonal_matrix(D, Q, v);
    matrix[D, Q] W_n;

    for (q in 1:Q)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(eigen_roots);
}


