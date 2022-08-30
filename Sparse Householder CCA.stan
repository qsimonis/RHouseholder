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
    int<lower=1> N; 
    //Number of observations for our two datasets
    int<lower = 1> D_1; 
    //Dimensionality of the observations for the first dataset
    int<lower = 1> D_2; 
    //Dimensionality of the observations for the second dataset
    int<lower=1> K_1; 
    //Dimensionality of the latent variable for the first view
    int<lower=1> K_2; 
    //Dimensionality of the latent variable for the second view
    matrix[D_1 + D_2, N] Y; 
    // Our concatenated dataset
}
transformed data{
  int<lower = 2> D = D_1 + D_2;
  int<lower = 2> K = K_1 + K_2;
}
parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*D - D*(D-1)/2] v;
    positive_ordered[D] eigenvalues;
    vector[K]<lower = 0, upper = 1> column_beta;
    vector[K]<lower = 0, upper = 1> alpha;
    array[D, K] B; 
    //The loading matrix for the specific views
    real<lower=0> eigenvalues_noise;
    real<lower = 0> eigenvalue_variance;
    vector[D] eigenvalue_differences;
    real<lower = 0> tau_1; 
    // View 1 noise
    real<lower = 0> tau_2; 
    // View 2 noise
}
transformed parameters{
    cholesky_factor_cov[D] L;
    vector[K_1] column_variance_1;
    vector[K_2] column_variance_2;
    vector[D - 2] max_eigenvalue;
    vector[D] eigenvalue_full_variance;
    
    {
        matrix[D, D] W = orthogonal_matrix(D, D, v);
        matrix[D, D] J;
        
        J = W*diag_matrix(eigenvalues)*W';
        for (d in 1:D)
            J[d, d] = J[d,d] + square(eigenvalues_noise) + 1e-14;
        L = cholesky_decompose(J);
    }
    for(i in 1:K){
      column_variance_1[i] = column_beta[i]/alpha[i];
      column_variance_2[i] = (1 - column_beta[i])/alpha[i];
    }
    for(d in 4:D){
      max_eigenvalue[d] = max(eigenvalues[1:d] - eigenvalues[2:d-1]);
    }
}
model{
    // Prior needed for orthogonal matrix distribution
    eigenvalues_noise ~ cauchy(0,1);
    v ~ normal(0,1);
    
    // Hyperpriors
    tau_1 ~ gamma(10^(-14), 10^(-14));
    tau_2 ~ gamma(10^(-14), 10^(-14));
    
    
    // Prior on eigenvalues
    log(eigenvalues[1]) ~ normal(0,eigenvalue_variance);
    log(eigenvalues[2]) ~ normal(0,eigenvalue_variance/log(eigenvalues[1]));
    for(i in 3:(D)){
      eigenvalues[i] ~ normal(0, eigenvalue_variance/(log(eigenvalues[i - 1]) - log(eigenvalues[i - 2])));
    }
    
    // Adding the log-Jacobian correction term:
    for(j in 1:D){
      target += -log(eigenvalues[j]);
    }
    
    
    
    for(j in 1:K){
      column_beta[j] ~ beta(.001,.001);
      alpha[j] ~ gamma(10^(-14), 10^(-14));
    }

    
    for(i in 1:D){
      for(j in 1:K){
        if(i <= D_1){
          B[i,j] ~ normal(0, column_variance_1[j]);
        }
        else{
          B[i,j] ~ normal(0, column_variance_2[j]);
        }
      }
    }
    
    for(j in 1:N){
      X ~ normal(0,1);
      Y[,j] ~ multi_normal_cholesky(B*X,L);
    }
}
generated quantities {
    matrix[D, D] U_n = orthogonal_matrix(D, D, v);
    matrix[D, D] W_n;
    
    for (q in 1:D)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(eigenvalues);
}
