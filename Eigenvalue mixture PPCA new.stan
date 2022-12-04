functions{
    matrix V_low_tri_plus_diag (int D, vector v) {
        // Put parameters into lower triangular matrix
        matrix[D, D] V;

        int idx = 1;
        for (d in 1:D) {
            for (q in 1:D) {
                if (d >= q) {
                    V[d, q] = v[idx];
                    idx += 1;
                } else
                V[d, q] = 0;
            }
        }
        for (q in 1:D){
            V[,q] = V[,q]/sqrt( sum(square(V[,q]) ) );
        }
        return V;
    }
    real sign(real x){
        if (x < 0.0)
            return -1.0;
        else
            return 1.0;
    }
    real indicator(real x, real a){
      if (x > a)
          return 1;
      else
          return 0;
    }
    matrix Householder (int k, matrix V) {
        // Householder transformation corresponding to kth column of V
        int D = rows(V);
        vector[D] v = V[, k];
        matrix[D,D] H;
        real sgn = sign(v[k]);
        
        v[k] +=  sgn; //v[k]/fabs(v[k]);
        H = diag_matrix(rep_vector(1, D)) - (2.0 / dot_self(v)) * (v * v');
        H[k:, k:] = -sgn*H[k:, k:];
        return H;
    }
    matrix[] H_prod_right (matrix V) {
        // Compute products of Householder transformations from the right, i.e. backwards
        int D = rows(V);
        matrix[D, D] H_prod[D + 1];
        H_prod[1] = diag_matrix(rep_vector(1, D));
        for (q in 1:D)
            H_prod[q + 1] = Householder(D - q + 1, V) * H_prod[q];
        return H_prod;    
    }
    matrix orthogonal_matrix (int D, vector v) {
        matrix[D, D] V = V_low_tri_plus_diag(D, v);
        // Apply Householder transformations from right
        matrix[D, D] H_prod[D + 1] = H_prod_right(V);
        return H_prod[D + 1][, 1:D];    
    }
}
data{
    int<lower=0> N; //Number of observations
    int<lower=1> D; //Dimension of observations
    int<lower=1> K; //Number of eigenvalue clusters
    vector[D] Y[N]; //Observations
}

parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*D - D*(D-1)/2] v;
    vector<lower = 0>[D-1] eigen_differences;
    real<lower = 0> first_eigenvalue;
    vector<lower = 0> [D-1] eigen_means[K];
    real<lower = 0> covariance_noise;
    real<lower = 0> eigen_variance;
    vector[D] mu;

}
transformed parameters{
    matrix[D, D] W;
    cholesky_factor_cov[D] L;
    vector[D] eigen_roots;
    vector[K] soft_z[D-1]; // log unnormalized clusters
    real neg_log_K;
    neg_log_K = -log(K);
    
    eigen_roots[1] = first_eigenvalue;
    
    {
      for(i in 2:D){
        eigen_roots[i] = exp(log(first_eigenvalue) - sum(eigen_differences[1:(i - 1)]));
      }
    }
    
    for (d in 1:(D-1)) {
    for (k in 1:K) {
      soft_z[d, k] = neg_log_K
                     - 0.5 * dot_self(eigen_means[k] - eigen_differences[d])/eigen_variance;
    }
  }



    {
        matrix[D, D] U = orthogonal_matrix(D, v);
        matrix[D, D] Q;

        W = U*diag_matrix(eigen_roots);

        Q = W*W';
        for (d in 1:D){
            Q[d, d] = Q[d,d] + covariance_noise + 1e-14;
        }
        L = cholesky_decompose(Q);
    }
    print("Estimated eigendifferences: ", eigen_differences);
    print("Estimated eigenvariance: ", eigen_variance);
}
model{
    mu ~ normal(.5,.1);
    covariance_noise ~ normal(0,.05);
    eigen_variance ~ normal(0,.15);
    
    //eigen_means_final[1] ~ normal(0,.1);
    //eigen_means_final[2] ~ normal(0,1);
    
    
    first_eigenvalue ~ normal(4.5,.1);
    
    eigen_means[1] ~ normal(.1,.1);
    eigen_means[2] ~ normal(1,.1);
    
    for (d in 1:(D-1)) {
    target += log_sum_exp(soft_z[d]);
  }

    v ~ normal(0,1);

    //prior on singular values
    
    Y ~ multi_normal_cholesky(mu, L);
}

generated quantities {
    matrix[D, D] U_n = orthogonal_matrix(D, v);
    matrix[D, D] W_n;

    for (q in 1:D)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(eigen_roots);
}