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
    vector eigenvalue_converter(vector v){
      int n = num_elements(v);
      vector[n] eigenvalues;
      for(i in 1:n){
        for(j in 1:i){
          eigenvalues[i] = exp(v[1] - sum(v[2:j]));
        }
      }
      return eigenvalues;
    }
}
data{
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> Q; //Q = D for our model
    int<lower = 1> D_1; //Dimensionality of the observations for the first dataset
    int<lower = 1> D_2; //Dimensionality of the observations for the second dataset
    int<lower = 1> K_1;
    int<lower = 1> K_2;
    vector[D] Y[N];
}
parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    positive_ordered[Q] sigma;
    
    vector[D] mu;
    real<lower = 0> tau_1;
    real<lower = 0> tau_2;
    real<lower = 0> eigen_variance;
    matrix[D , K_1 + K_2] B;
    vector[K_1 + K_2] beta;
    vector[K_1 + K_2] alpha;
    vector[K_1 + K_2] X;
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    vector<lower = 0>[Q] eigen_differences;
    {
      eigen_differences[1] = log(sigma[1]);
      for(q in 2:Q){
        eigen_differences[q] = log(sigma[q - 1]) - log(sigma[q]);
      }
    }
    vector[Q] eigen_max;
    {
      for(q in 1:Q){
        eigen_max[q] = max(eigen_differences[1:q]);
      }
    }
    
    {
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;
        
        W = U*diag_matrix(sqrt(eigenvalue_converter(eigen_differences)));
        
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
    mu ~ normal(0, 10);
    tau_1 ~ cauchy(0,1);
    tau_2 ~ cauchy(0,1);
    beta ~ beta(.001,.001);
    alpha ~ cauchy(0,1);
    
    for(d in 1:(D_1 + D_2)){
      for(k in 1:(K_1 + K_2)){
        if(k <= K_1){
          B[d,k] ~ normal(0,beta[k]/alpha[k]);
        }
        else{
          B[d,k] ~ normal(0, (1- beta[k])/alpha[k]);
        }
      }
    }
    
    v ~ normal(0,1);
    
    //prior on sigma
    eigen_differences[1] ~ normal(0, eigen_variance);
    eigen_differences[2] ~ normal(0, eigen_variance/eigen_differences[1]);
    eigen_differences[3] ~ normal(0, eigen_variance/(eigen_differences[2]));
    for(q in 4:Q){
      eigen_differences[q] ~ normal(0, eigen_variance*(eigen_max[q]/eigen_differences[q - 1]));
    }
    target += -sum(log(sigma));
    
    X ~ normal(0,1);
    Y ~ multi_normal_cholesky(B*X, L);   
}
generated quantities {
    matrix[D, Q] U_n = orthogonal_matrix(D, Q, v);
    matrix[D, Q] W_n;
    
    for (q in 1:Q)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(sqrt(eigenvalue_converter(eigen_differences)));
}
