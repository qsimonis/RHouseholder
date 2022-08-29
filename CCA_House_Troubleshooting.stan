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
    vector<lower = 0, upper = 1>[Q] local_variance;
    real<lower = 0> tau_1;
    real<lower = 0> tau_2;
    real<lower = 0> eigen_variance;
    matrix[D , K_1 + K_2] B;
    vector<lower = 0, upper = 1>[K_1 + K_2] beta_column;
    vector<lower = 0>[K_1 + K_2] alpha;
    vector[K_1 + K_2] X;
}
transformed parameters{
    matrix[D, Q] W;
    cholesky_factor_cov[D] L;
    vector<lower = 0>[Q] eigen_differences;
    vector[Q] eigen_roots_corrected;
    vector[Q] eigen_max;
    vector[Q] relative_min;
    {
      for(i in 1:Q){
        eigen_roots_corrected[i] = eigen_roots[(Q - i) + 1];
      }
    }
    {
      eigen_differences[1] = 2*log(eigen_roots_corrected[1]);
      print("1:", "eigen_differences[1]", eigen_differences[1]);
      print("q", 1, "eigen_roots_corrected[1]", eigen_roots_corrected[1]);
      for(q in 2:Q){
        eigen_differences[q] = 2*log(eigen_roots_corrected[q-1]+.01) - 2*log(eigen_roots_corrected[q] + .01);
        print("q", q, "eigen_roots_corrected[q]", eigen_roots_corrected[q]);
        print("q:", q,  "eigen_differences[q]", eigen_differences[q]);
      }
    }
    {
      eigen_max[1] = .01;
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
        matrix[D, Q] U = orthogonal_matrix(D, Q, v);
        matrix[D, D] K;

        W = U*diag_matrix(eigen_roots_corrected + 1);

        K = W*W';
        for (d in 1:D_1)
            K[d, d] = K[d,d] + square(tau_1) + .01;
        for (d in 1:D_2){
          K[d,d] = K[d,d] + square(tau_2) + .01;
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
      local_variance[q] ~ beta(eigen_max[q], relative_min[q]);
    }
    print(local_variance)
    for(d in 1:(D_1 + D_2)){
      for(k in 1:(K_1 + K_2)){
        if(k <= K_1){
          B[d,k] ~ normal(0,beta_column[k]/alpha[k]);
        }
        else{
          B[d,k] ~ normal(0, (1- beta_column[k])/alpha[k]);
        }
      }
    }

    v ~ normal(0,1);

    //prior on singular values
    eigen_roots_corrected[1] ~ chi_square(1);
    eigen_roots_corrected[2] ~ normal(0, eigen_variance*local_variance[1]);
    eigen_roots_corrected[3] ~ normal(0, eigen_variance*local_variance[2]);
    for(q in 4:Q){
      eigen_roots_corrected[q] ~ normal(0, eigen_variance*local_variance[q - 1]);
    }
    target += sum(2*log(eigen_roots));

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
    W_n = U_n*diag_matrix(eigen_roots_corrected);
}
