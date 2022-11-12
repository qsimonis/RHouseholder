functions{
    matrix V_low_tri_plus_diag (int D, int Q, vector v) {
        // Put parameters into lower triangular matrix
        matrix[D, Q] V;

        int idx = 1;
        for (d in 1:D) {
            for (q in 1:Q) {
                if (d >= q) {
                    V[d, q] = v[idx];
                    idx += 1;
                } else
                V[d, q] = 0;
            }
        }
        for (q in 1:Q){
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
        int Q = cols(V);
        matrix[D, D] H_prod[Q + 1];
        H_prod[1] = diag_matrix(rep_vector(1, D));
        for (q in 1:Q)
            H_prod[q + 1] = Householder(Q - q + 1, V) * H_prod[q];
        return H_prod;    
    }
    matrix orthogonal_matrix (int D, int Q, vector v) {
        matrix[D, Q] V = V_low_tri_plus_diag(D, Q, v);
        // Apply Householder transformations from right
        matrix[D, D] H_prod[Q + 1] = H_prod_right(V);
        return H_prod[Q + 1][, 1:Q];    
    }
}
data{
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> D_1;
    int<lower=1> D_2;
    int<lower = 1> K_1;
    int<lower = 1> K_2;
    int<lower=1> Q;
    vector[D] Y[N];
    real<lower = 0> eigenvalue_tuning_number;

}

parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    vector<lower = 0>[Q-1] eigen_differences;
    real<lower = 0> first_eigenvalue;
    positive_ordered[2] eigen_means;
    real<lower = 0> tau_1;
    real<lower = 0> tau_2;
    real<lower = 0> eigen_variance;
    vector[K_1 + K_2] X[N];
    vector<lower = 0>[K_1 + K_2] column_variances;
    vector[D_1*K_1 + D_2*K_2] view_vector;

}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    matrix[D, K_1 + K_2] view_matrix;
    vector[D] mu[N];
    vector[Q] eigen_roots;
    vector[Q-1] eigen_max;
    vector[Q-1] eigen_min;
    vector[2] corrected_eigen_means;
    vector[Q-1] mixture_proportions;
    vector[Q - 1] eigen_means_final;

    
    corrected_eigen_means[1] = eigen_means[2];
    corrected_eigen_means[2] = eigen_means[1];
    
    {
      for(q in 1:(Q-1)){
        eigen_max[q] = max(eigen_differences[1:q]);
      }
    }
    
    {
      for(q in 1:(Q-1)){
        eigen_min[q] = min(eigen_differences[1:q]) + .01;
      }
    }
    
    eigen_roots[1] = first_eigenvalue;
    
    {
      for(i in 2:Q){
        eigen_roots[i] = exp(log(first_eigenvalue) - sum(eigen_differences[1:(i - 1)]));
      }
    }
    
    for(i in 1:(Q - 1)){
      mixture_proportions[i] = indicator(eigen_max[i], eigenvalue_tuning_number);
    }
    
    for(i in 1:(Q-1)){
      eigen_means_final[i] = corrected_eigen_means[1] + mixture_proportions[i]*corrected_eigen_means[2];
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
    print("Estimated eigenvalues: ", eigen_roots);
    print("Estimated mixture component: ", mixture_proportions);
}
model{
    tau_1 ~ cauchy(0,1);
    tau_2 ~ cauchy(0,1);
    eigen_variance ~ normal(0,.05);
    
    first_eigenvalue ~ normal(10,1);
    for(q in 1:(Q-1)){
      eigen_differences[q] ~ normal(eigen_means_final[q], eigen_variance);
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
    
    Y ~ multi_normal_cholesky(mu, L);
    X ~ multi_normal_cholesky(rep_vector(0, K_1 + K_2), diag_matrix(rep_vector(1, K_1 + K_2)));
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

