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
    int<lower=0> N;
    int<lower=1> D;
    int<lower=1> D_1;
    int<lower=1> D_2;
    int<lower = 1> K_1;
    int<lower = 1> K_2;
    int<lower=1> Q;
    vector[D] Y[N];
}


parameters{
    //vector[D*(D+1)/2 + D] v;
    vector[D*Q - Q*(Q-1)/2] v;
    positive_ordered[Q] sigma;
    positive_ordered[Q] sigma_weight;
    real<lower = 0> weight_variance;
    real<lower=0> sigma_noise;
    real<lower = 0> view1_noise;
    real<lower = 0> view2_noise;
    vector[D_1*K_1 + D_2*K_2] view_vector;
    vector[K_1 + K_2] X[N];
    vector<lower = 0>[K_1 + K_2] column_variances;
    real<lower = 0> view1_noise_hyperprior;
    real<lower = 0> view2_noise_hyperprior;
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    positive_ordered[Q] sigma_new;
    matrix[D, K_1 + K_2] view_matrix;
    vector[D] mu[N];
    
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
    
    {
      for(i in 1:Q){
        sigma_new[i] = sigma[i]*sigma_weight[i];
      }
    }
    
    {
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;
        
        W = U*diag_matrix(sigma_new);
        K = W*W';
        for (d in 1:D){
          if(d <= D_1){
            K[d, d] = K[d,d] + square(view1_noise) + 1e-14;
          }
          else{
            K[d, d] = K[d,d] + square(view2_noise) + 1e-14;
          }
        }
        L = cholesky_decompose(K);
    }
    
    for(i in 1:N){
      mu[i] = view_matrix*X[i];
    }
    
    print("Estimated eigenvalues: ", sigma);
    print("Estimated weights: ", sigma_weight);
}
model{
    view1_noise ~ cauchy(0,view1_noise_hyperprior);
    view2_noise ~ cauchy(0,view2_noise_hyperprior);
    view1_noise_hyperprior ~ cauchy(0,1);
    view2_noise_hyperprior ~ cauchy(0,1);
    
    
    weight_variance ~ cauchy(0,1);
    for(i in 1:Q){
      sigma_weight[i] ~ cauchy(0,weight_variance);
    }
    
    
    v ~ normal(0,1);
    
    //prior on sigma
    target += -0.5*sum(square(sigma)) + (D-Q-1)*sum(log(sigma));
    for (i in 1:Q)
        for (j in (i+1):Q)
            target += log(square(sigma[Q-i+1]) - square(sigma[Q-j+1]));
    target += sum(log(2*sigma));
    
    for(j in 1:(K_1 + K_2)){
      column_variances[j] ~ cauchy(0,1);
    }
    
    for(i in 1:D){
      for(j in 1:(K_1 + K_2)){
        if(i <= D_1 || j <= K_1){
          view_vector[i*j] ~ normal(0,column_variances[j]);
        }
        else{
          view_vector[(i-D_1)*(j - K_1) + D_1*K_1] ~ normal(0, column_variances[j]);
        }
      }
    }
    
    Y ~ multi_normal_cholesky(mu, L);   
}
generated quantities {
    matrix[D, Q] U_n = orthogonal_matrix(D, Q, v);
    matrix[D, Q] W_n;
    
    for (q in 1:Q)
        if (U_n[1,q] < 0){
            U_n[,q] = -U_n[,q];
        }
    W_n = U_n*diag_matrix(sigma_new);
}