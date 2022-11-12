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
    int<lower=0> N;
    int<lower=1> D;
    vector[D] Y[N];
    real<lower = 0> eigenvalue_tuning_number;

}

parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*D - D*(D-1)/2] v;
    vector<lower = 0>[D-1] eigen_differences;
    real<lower = 0> first_eigenvalue;
    positive_ordered[2] eigen_means;
    real<lower = 0> covariance_noise;
    real<lower = 0> eigen_variance;
    vector[D] mu;

}
transformed parameters{
    matrix[D, D] W;
    cholesky_factor_cov[D] L;
    vector[D] eigen_roots;
    vector[D-1] eigen_max;
    vector[D-1] eigen_min;
    vector[2] corrected_eigen_means;
    vector[D-1] mixture_proportions;
    vector[D - 1] eigen_means_final;

    
    corrected_eigen_means[1] = eigen_means[2];
    corrected_eigen_means[2] = eigen_means[1];
    
    {
      for(q in 1:(D-1)){
        eigen_max[q] = max(eigen_differences[1:q]);
      }
    }
    
    {
      for(q in 1:(D-1)){
        eigen_min[q] = min(eigen_differences[1:q]) + .01;
      }
    }
    
    eigen_roots[1] = first_eigenvalue;
    
    {
      for(i in 2:D){
        eigen_roots[i] = exp(log(first_eigenvalue) - sum(eigen_differences[1:(i - 1)]));
      }
    }
    
    for(i in 1:(D - 1)){
      mixture_proportions[i] = indicator(eigen_max[i], eigenvalue_tuning_number);
    }
    
    for(i in 1:(D-1)){
      eigen_means_final[i] = corrected_eigen_means[1] + mixture_proportions[i]*corrected_eigen_means[2];
    }

    {
        matrix[D, D] U = orthogonal_matrix(D, v);
        matrix[D, D] K;

        W = U*diag_matrix(sqrt(eigen_roots));

        K = W*W';
        for (d in 1:D){
            K[d, d] = K[d,d] + square(covariance_noise) + 1e-14;
        }
        L = cholesky_decompose(K);
    }
    print("Estimated eigenvalues: ", eigen_roots);
    print("Estimated mixture component: ", mixture_proportions);
}
model{
    mu ~ normal(0,10);
    covariance_noise ~ normal(0,1);
    eigen_variance ~ normal(0,.25);
    
    //eigen_means_final[1] ~ normal(0,.1);
    //eigen_means_final[2] ~ normal(0,1);
    
    
    first_eigenvalue ~ normal(0,2);
    for(q in 1:(D-1)){
      eigen_differences[q] ~ normal(eigen_means_final[q], eigen_variance);
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

