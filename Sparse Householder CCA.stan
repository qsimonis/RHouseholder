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
    int<lower=1> N; #Number of observations for our two datasets
    int<lower = 1> D_1; #Dimensionality of the observations for the first dataset
    int<lower = 1> D_2; #Dimensionality of the observations for the second dataset
    int<lower=1> K_1; #Dimensionality of the latent variable for the first view
    int<lower=1> K_2; #Dimensionality of the latent variable for the second view
    vector[D_1 + D_2] Y[N]; # Our concatenated dataset
}
parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[(D_1 + D_2)*(D_1 + D_2) - (D_1 + D_2)*((D_1 + D_2)-1)/2] v; # THIS nay need to be changed.
                                                          # It describes the shared variability.
    positive_ordered[D_1 + D_2] sigma;
    real<lower = 0> bernoulli_hyperparameter;
    vector[D_1 + D_2] bernoulli_parameter;
    vector[K_1 + K_2] alpha;
    matrix[D_1 + D_2, K_1 + K_2] B; #The loading matrix for the specific views
    real<lower=0> sigma_noise;
    real<lower = 0> eigenvalue_variance;
    real<lower = 0> tau_1; # View 1 noise
    real<lower = 0> tau_2; # View 2 noise
    vector[K_1 + K_2] X[N]; # View-specific latent variable
}
transformed parameters{
    cholesky_factor_cov[D_1 + D_2] L;
    
    {
        matrix[D_1 + D_2, D_1 + D_2] W = orthogonal_matrix(D_1 + D_2, D_1 + D_2, v);
        matrix[D_1 + D_2, D_1 + D_2] K;
        
        K = W*diag_matrix(sigma)*W';
        for (d in 1:D)
            K[d, d] = K[d,d] + square(sigma_noise) + 1e-14;
        L = cholesky_decompose(K);
    }
}
model{
    
    sigma_noise ~ cauchy(0,1);
    v ~ normal(0,1);
    
    # Prior on sigma
    log(sigma[1]) ~ normal(0,eigenvalue_variance)
    log(sigma[2]) ~ normal(0,eigenvalue_variance/log(sigma[1]))
    for(i in 3:(D_1 + D_2)){
      sigma[i] ~ normal(0, eigenvalue_variance/(log(sigma[i - 1]) - log(sigma[i - 2])))
    }
    
    # Adding the log-Jacobian correction term:
    for(j in 1:(D_1 + D_2)){
      target += -log(sigma[j])
    }
    
    bernoulli_hyperparameter ~ beta(.001,.001)
    
    for(j in (K_1 + K_2)){
      bernoulli_parameter[j] ~ bernoulli(bernoulli_hyperparameter)
    }
    
    for(j in (K_1 + K_2)){
      alpha[j] ~ gamma(10^(-14), 10^(-14)) # THIS MAY NEED TO BE CHANGED
    }
    
    tau_1 ~ gamma(10^(-14), 10^(-14))
    
    tau_2 ~ gamma(10^(-14), 10^(-14))

    # Note here that the choice of the hyperparameters for the gamma were chosen to be:
    # alpha_0 = beta_0 = 10^(-14)
    
    
    for(i in (D_1 + D_2)){
      for(j in (K_1 + K_2)){
        if(i <= D_1){
          B[i,j] ~ bernoulli_parameter[j]*normal(0, 1/alpha[j])
        }
        else{
          B[i,j] ~ (1 - bernoulli_parameter[j])*normal(0, 1/alpha[j])
        }
      }
    }
    
    X ~ normal(0,1)
    Y ~ multi_normal_cholesky(B*X, L);   
}
generated quantities {
    matrix[N, D_1 + D_2] U_n = orthogonal_matrix(N, D_1 + D_2, v);
    matrix[N, D_1 + D_2] W_n;
    
    for (q in 1:Q)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(sigma);
}
