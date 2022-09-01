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
      eigenvalues[1] = exp(v[1]);
      for(i in 2:n){
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
    positive_ordered[Q] eigen_roots;
    vector<lower = 0, upper = 1>[Q] local_eigen_variance;
    real<lower = 0> tau_1;
    real<lower = 0> tau_2;
    real<lower = 0> eigen_variance;
    matrix[D , K_1 + K_2] B;
    vector<lower = 0>[K_1 + K_2] column_tau;
    vector<lower = 0, upper = 1>[K_1 + K_2] column_view_lambda;
    vector<lower = 0, upper = 1>[(K_1 + K_2)] column_view_alpha;
    vector<lower = 0>[K_1 + K_2] alpha;
    vector[K_1 + K_2] X[N];
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    vector<lower = 0>[D] eigen_differences;
    vector[D] eigen_roots_corrected;
    vector[D] eigen_max;
    vector[D] relative_min;
    vector<lower = 0>[K_1 + K_2] column_view_beta;
    vector<lower = 0>[(K_1 + K_2)] column_variance_1;
    vector<lower = 0>[(K_1 + K_2)] column_variance_2;
    vector[D] mu[N];
    {
      for(i in 1:D){
        eigen_roots_corrected[i] = eigen_roots[(D - i) + 1];
      }
    }
    {
      for(i in 1:(K_1 + K_2)){
        column_view_beta[i] = 1 - column_view_alpha[i];
        for(j in 1:(D_1 + D_2)){
          if(j <= D_1){
            column_variance_1[i] = (column_view_lambda[i])/column_tau[i];
          }
          else{
            column_variance_2[i] = (1 - column_view_lambda[i])/column_tau[i];
          }
      }
      }
    }
    {
      eigen_differences[1] = 2*log(eigen_roots_corrected[1]);
      for(d in 2:D){
        eigen_differences[d] = 2*log(eigen_roots_corrected[d-1]+.01) - 2*log(eigen_roots_corrected[d] + .01);
      }
    }
    {
      eigen_max[1] = eigen_differences[1];
      for(d in 2:D){
        eigen_max[d] = max(eigen_differences[2:d]);
      }
    }
    {
      relative_min[1] = eigen_differences[1];
      for(d in 2:D){
        relative_min[d] = min(eigen_differences[1:D]);
      }
    }

    {
        matrix[D, D] U = orthogonal_matrix(D, D, v);
        matrix[D, D] K;

        W = U*diag_matrix(eigen_roots_corrected);

        K = W*W';
        for (d in 1:D_1){
            K[d, d] = K[d,d] + square(tau_1) + .01;
        }
        for (d in 1:D_2){
          K[d,d] = K[d,d] + square(tau_2) + .01;
        }
        L = cholesky_decompose(K);
    }
    {
      for(i in 1:N){
        mu[i] = B*X[i];
      }
    }
    print("The estimated singular values: ", eigen_roots_corrected);
    print("The estimated maximum eigen gaps: ", eigen_max);
    print("The estimated minimum eigen gaps: ", relative_min);
    print("The estimated local column variances: ")
    print("The estimated global column variances: ", column_tau[1:(K_1 + K_2)]);
    print("The estimated noise variances: ", tau_1, " ", tau_2)
}
model{
    tau_1 ~ exponential(30);
    tau_2 ~ exponential(30);
    column_tau ~ cauchy(0,1);
    column_view_alpha ~ beta(.005,.005);
    column_view_lambda ~ beta(column_view_alpha, column_view_beta);
    eigen_variance ~ cauchy(0,1);
    for(d in 1:D){
      local_eigen_variance[d] ~ beta(max(eigen_differences[1:d])/min(eigen_differences[1:d]), min(eigen_differences[1:d])/max(eigen_differences[1:d]));
    }
    for(d in 1:(D_1 + D_2)){
      for(k in 1:(K_1 + K_2)){
        if(d <= D_1){
          B[d,k] ~ normal(0,column_variance_1[k]);
        }
        else{
          B[d,k] ~ normal(0, column_variance_2[k]);
        }
      }
    }

    v ~ normal(0,1);

    //prior on singular values
    eigen_roots_corrected[1] ~ uniform(0,100);
    for(q in 2:Q){
      eigen_roots_corrected[q] ~ normal(0, local_eigen_variance[q - 1]/eigen_variance);
    }
    target += sum(2*log(eigen_roots));
    
    for(i in 1:N){
      X[i] ~ multi_normal(rep_vector(0, K_1 + K_2),diag_matrix(rep_vector(1,K_1 + K_2)));
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
    W_n = U_n*diag_matrix(eigen_roots_corrected);
}
