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
    matrix[D_1, K_1] B_1;
    matrix[D_2, K_2] B_2;
    vector<lower = 0>[K_1 + K_2] column_tau;
    vector[K_1 + K_2] X[N];
}

transformed parameters{
    matrix[D, K_1 + K_2] partial_matrix;
    vector[D] mu[N];
    {
      for(j in 1:(K_1 + K_2)){
        for(i in 1:D){
          if((j > K_1 && i <= D_1) || (j <= K_1 && i > D_1)){
            partial_matrix[i,j] = 0;
          }
          else if (j <= K_1 && i <= D_1){
            partial_matrix[i,j] = B_1[i,j];
          }
          else if (j > K_1 && i > D_1){
            partial_matrix[i,j] = B_2[i - D_1, j - K_1];
          }
        }
      }
    }
    
    {
      for(i in 1:N){
        mu[i] = partial_matrix*X[i];
      }
    }
    
    print("The estimated global column variances: ", column_tau[1:(K_1 + K_2)]);
    print("The estimated observation mean", mu[1]);
}

model{
    for(d in 1:(D_1 + D_2)){
      for(k in 1:(K_1 + K_2)){
        if((d <= D_1 && k <= K_1) || (d > D_1 && k > K_1)){
          column_tau[k] ~ cauchy(0,1);
          partial_matrix[d,k] ~ normal(0,column_tau[k]);
        }
      }
    }

    //prior on singular values
    
    for(i in 1:N){
      X[i] ~ multi_normal(rep_vector(0, K_1 + K_2),diag_matrix(rep_vector(1,K_1 + K_2)));
    }
    
    for(i in 1:N){
      Y[i] ~ multi_normal_cholesky(mu[i], diag_matrix(rep_vector(.1,D_1 + D_2)));
    }
}

