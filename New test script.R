require(rstiefel) # This will allow the easy generation of random rectangular orthogonal matrices via haar measure over the space of orthogonal matrices

require(mvtnorm) # This will be used for generating multivariate normal observations for our artificial datasets

require(extraDistr)

eigenvalue.simulator <- function(dims, alpha, beta){
  eigenvalues.simulated <- sort(rgamma(dims, shape = alpha, rate = beta), decreasing = T)
  return(eigenvalues.simulated)
}

sparse.eigenvalue.simulator <- function(eigenvalues, proportion, max){
  number.to.shrink <- floor(length(eigenvalues)*proportion) - 1
  eigenvalues[(length(eigenvalues) - number.to.shrink):length(eigenvalues)] <- .3*sort(runif( n = number.to.shrink + 1, min = 0, max = max), decreasing = T)
  return(eigenvalues)
}

shared.covariance.generator.eigen <- function(eigenvalues){
  Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = length(eigenvalues), R = length(eigenvalues))
  Generated.covariance <- Generated.orthogonal.matrix%*%diag(eigenvalues)
  return(Generated.covariance)
}

noise.covariance.generator <- function(data1.dim, data2.dim, shared.noise.1, shared.noise.2){
  covariance.noise <- matrix(0, nrow = data1.dim + data2.dim, ncol = data1.dim + data2.dim)
  noise.1 <- shared.noise.1
  noise.2 <- shared.noise.2
  for(i in 1:data1.dim){
    covariance.noise[i,i] <- noise.1
  }
  for(i in (data1.dim + 1):(data1.dim + data2.dim)){
    covariance.noise[i,i] <- noise.2
  }
  return(covariance.noise)
}

column.variance.generator <- function(view1.dim, view2.dim){
  column.variances <- rhcauchy(view1.dim + view2.dim)
  return(column.variances)
}

view.specific.matrix.generator <- function(view1.dim, view2.dim, data1.dim, data2.dim,
                                           column.variances){
  require(mvtnorm)
  view.matrix <- matrix(0, nrow = data1.dim + data2.dim, ncol = view1.dim + view2.dim)
  for(j in 1:(view1.dim + view2.dim)){
    view.matrix[,j] <- as.vector(rmvnorm(1, mean = rep(0, data1.dim + data2.dim),
                                 sigma = diag(rep(column.variances[j], data1.dim + data2.dim), nrow = data1.dim + data2.dim)))
    for(i in 1:(data1.dim + data2.dim)){
      if(i > data1.dim && j <= view1.dim){
        view.matrix[i,j] = 0
      }
      else if(i <= data1.dim && j > view1.dim){
        view.matrix[i,j] = 0
      }
  }
  }
  return(view.matrix)
}

set.seed(12104)

shared.eigenvalues <- eigenvalue.simulator(dims = 10, alpha = 1.75, beta = 1)
sparse.eigenvalues <- sparse.eigenvalue.simulator(eigenvalues = shared.eigenvalues, proportion = .5, max = .1)
B <- shared.covariance.generator.eigen(sparse.eigenvalues)
view.noise <- noise.covariance.generator(data1.dim = 4, data2.dim = 6, shared.noise.1 = .3, shared.noise.2 = .5)
column.variances <- sort(column.variance.generator(view1.dim = 2, view2.dim = 3), decreasing = F)
view.matrix <- view.specific.matrix.generator(view1.dim = 2, view2.dim = 3, data1.dim = 4, data2.dim = 6, column.variances = column.variances)

Y <- B%*%t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)))) + 
  t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)) + view.noise))


data = list(N=n, D=d, K=k, ones=1, 
            y=Y)


library(rstan)
n = 300
D1 = 4
D2 = 6
K1 = 2
K2 = 3
k = K1 + K2
d = D1 + D2

max.eigen = max(sparse.eigenvalues)
uni.max = max.eigen + runif(1, min = -.3, max = .3)
set.seed(4333213)
chain.1 = list(sigma_weight = sort(rbeta((d), shape1 = .1, shape2 = .1), decreasing = F),
                      sigma = sort(c(runif(d/2 , min = 0, max = 1),
                                     uni.max*runif(d/2 , min = .25 , max = 1)), decreasing = F),
               column.variances = sort(rhcauchy(K1 + K2), decreasing = F))
chain.2 = list(sigma_weight = chain.1$sigma_weight,
               sigma = sort(chain.1$sigma + runif(d, min = - min(chain.1$sigma), max = .5), decreasing = F),
               column.variances = sort(chain.1$column.variances + runif(K1 + K2, min = - min(chain.1$column.variances), max = .25),
                                    decreasing = F))
chain.3 = list(sigma_weight = chain.1$sigma_weight,
               sigma = sort(chain.1$sigma + runif(d, min = - min(chain.1$sigma), max = .5), decreasing = F),
               column.variances = sort(chain.1$column.variances + runif(K1 + K2, min = - min(chain.1$column.variances), max = .25),
                                       decreasing = F))
chain.4 = list(sigma_weight = chain.1$sigma_weight,
               sigma = sort(chain.1$sigma + runif(d, min = - min(chain.1$sigma), max = .5), decreasing = F),
               column.variances = sort(chain.1$column.variances + runif(K1 + K2, min = - min(chain.1$column.variances), max = .25),
                                       decreasing = F))

names(chain.1) = c("sigma_weight", "sigma", "column_variances")
names(chain.2) = c("sigma_weight", "sigma", "column_variances")
names(chain.3) = c("sigma_weight", "sigma", "column_variances")
names(chain.4) = c("sigma_weight", "sigma", "column_variances")
init.list = list(chain.1, chain.2, chain.3, chain.4)
names(init.list) = c("chain 1","chain 2", "chain 3", "chain 4")

Y.new <- view.matrix %*% t(rmvnorm(n = n, mean = rep(0, ncol(view.matrix)), sigma = diag(ncol(view.matrix)))) + 
  B%*%t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)))) + 
  t(rmvnorm(n = n, mean = rep(0, ncol(B)), sigma = diag(ncol(B)) + view.noise))

CCA.data <- list(
  N = n,
  Y = t(Y.new),
  D = d,
  D_1 = 4,
  D_2 = 6,
  K_1 = 2,
  K_2 = 3,
  Q = d
)

sparse.eigenvalues
column.variances

fit.householder.CCA.desktop <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/PPCA_House_Test_ARD_Extended.stan",
                                    data = CCA.data, seed = 4,
                                    control = list(max_treedepth = 13, adapt_delta = .3),
                                    init = init.list, iter = 1000,
                                    thin = 10,
                                    cores = parallel::detectCores())

library("bayesplot")
library("ggplot2")
library("rstanarm")
posterior.CCA <- as.array(fit.householder.CCA.desktop)
color_scheme_set("red")
mcmc_intervals(posterior.CCA, pars = c("sigma_new"))


summary(fit.householder.CCA.desktop, pars = c("sigma"))$summary
summary(fit.householder.CCA.desktop, pars = c("sigma_weight"))$summary
summary(fit.householder.CCA.desktop, pars = c("sigma_new"))$summary
summary(fit.householder.CCA.desktop, pars = c("view1_noise"))$summary
summary(fit.householder.CCA.desktop, pars = c("view2_noise"))$summary
summary(fit.householder.CCA.desktop, pars = c("view_matrix"))$summary
summary(fit.householder.CCA.desktop, pars = c("column_variances"))$summary




library(ggplot2)

MCMC.data.plot <- data.frame(summary(fit.householder.CCA.desktop, pars = c("sigma_new"))$summary)
sigma_vec <- c("sigma_1","sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7",
               "sigma_8", "sigma_9", "sigma_10")
MCMC.data.plot$id <- c(MCMC.data.plot$id, levels=sigma_vec)
MCMC.data.plot$id <- factor(MCMC.data.plot$id, levels=sigma_vec)
colors <- c("True Value" = "red")
ggplot(MCMC.data.plot,aes(x=id)) +
   geom_boxplot(aes(lower=mean -2*sd,upper=mean +2*sd,middle=mean ,ymin=mean - 3*sd,ymax=mean +3*sd),
                stat="identity") + 
  geom_point(data = data.frame(x = factor(sigma_vec, levels = sigma_vec), y = sort(sparse.eigenvalues,
                                                                                   decreasing = F)),
             aes(x=x, y=y,
             color = "True Value")) +
  labs(title="Estimated Singular Values",
       x ="Index", y = "Estimated Value", color = "Legend") + 
  scale_color_manual(values = colors) +
 theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# ggplot code 
ggplot(aes(y = percent, x = factor(sparse.eigenvalues)), data = mydata)+
  ggtitle("Boxplot con media, 95%CI, valore min. e max.")+xlab("Singular Value")+ylab("Valori")+
  stat_summary(fun.data = min.mean.sd.max, geom = "boxplot")+
  geom_jitter(position=position_jitter(width=.2), size=3) 




fit.householder.CCA <- stan(file = "D:/School/Projects/GitMCMCHouseholder/RHouseholder/PPCA_House_Test_ARD_Extended.stan",
                            data = CCA.data, chains = 1, seed = 1234,
                            control = list(max_treedepth = 12),
                            init = init.list, iter = 300)

summary(fit.householder.CCA, pars = c("sigma"))$summary
summary(fit.householder.CCA, pars = c("sigma_weight"))$summary
summary(fit.householder.CCA, pars = c("sigma_new"))$summary
summary(fit.householder.CCA, pars = c("view1_noise"))$summary
summary(fit.householder.CCA, pars = c("view2_noise"))$summary
summary(fit.householder.CCA, pars = c("view_matrix"))$summary





fit.householder <- stan(file = "D:/School/Projects/GitMCMCHouseholder/RHouseholder/PPCA_House_Test.stan", data = data.house, chains = 1, init = init.list, iter = 300)

summary(fit.householder, pars = c("sigma_new"))$summary

summary(fit.householder, pars = c("W"))$summary



###### Starting a new test script at the bottom here

#This function  will generate a matrix with random normal entries

wishart.matrix.generator <- function(n.row, n.col){
  generated.matrix <- matrix(0, nrow = n.row, ncol = n.col)
  for(i in 1:nrow(generated.matrix)){
    for(j in 1:ncol(generated.matrix)){
      generated.matrix[i,j] = rnorm(1)
    }
  }
  return(generated.matrix %*% t(generated.matrix))
}

sparse.eigenvalue.simulator <- function(eigenvalues, proportion, max, shrinkage){
  number.to.shrink <- floor(length(eigenvalues)*proportion) - 1
  eigenvalues[(length(eigenvalues) - number.to.shrink):length(eigenvalues)] <- shrinkage*sort(runif( n = number.to.shrink + 1, min = 0, max = max), decreasing = T)
  return(eigenvalues)
}

shared.covariance.generator.eigen <- function(eigenvalues){
  Generated.orthogonal.matrix <-  rstiefel::rustiefel(m = length(eigenvalues), R = length(eigenvalues))
  Generated.covariance <- Generated.orthogonal.matrix%*%diag(eigenvalues)
  return(Generated.covariance)
}

noise.covariance.generator <- function(data1.dim, data2.dim, shared.noise.1, shared.noise.2){
  covariance.noise <- matrix(0, nrow = data1.dim + data2.dim, ncol = data1.dim + data2.dim)
  noise.1 <- shared.noise.1
  noise.2 <- shared.noise.2
  for(i in 1:data1.dim){
    covariance.noise[i,i] <- noise.1
  }
  for(i in (data1.dim + 1):(data1.dim + data2.dim)){
    covariance.noise[i,i] <- noise.2
  }
  return(covariance.noise)
}

column.variance.generator <- function(view1.dim, view2.dim){
  column.variances <- extraDistr::rhcauchy(view1.dim + view2.dim)
  return(column.variances)
}

view.specific.matrix.generator <- function(view1.dim, view2.dim, data1.dim, data2.dim,
                                           column.variances){
  require(mvtnorm)
  view.matrix <- matrix(0, nrow = data1.dim + data2.dim, ncol = view1.dim + view2.dim)
  for(j in 1:(view1.dim + view2.dim)){
    view.matrix[,j] <- as.vector(rmvnorm(1, mean = rep(0, data1.dim + data2.dim),
                                         sigma = diag(rep(column.variances[j], data1.dim + data2.dim), nrow = data1.dim + data2.dim)))
    for(i in 1:(data1.dim + data2.dim)){
      if(i > data1.dim && j <= view1.dim){
        view.matrix[i,j] = 0
      }
      else if(i <= data1.dim && j > view1.dim){
        view.matrix[i,j] = 0
      }
    }
  }
  return(view.matrix)
}

CCA.normal.data.generator <- function(n, D1, D2, K1, K2, prop.CCA){
  k = K1 + K2
  d = D1 + D2
  view.specific.column.variances <- column.variance.generator(K1, K2)
  view.specific.matrix <- view.specific.matrix.generator(view1.dim = K1, view2.dim = K2,
                                                         data1.dim = D1, data2.dim = D2,
                                                         column.variances = view.specific.column.variances)
  wishart.matrix <- wishart.matrix.generator(n.row = d, n.col = d)
  wishart.eigenvalues <- sqrt(eigen(wishart.matrix)$values)
  sparse.wishart.eigenvalues <- sparse.eigenvalue.simulator(wishart.eigenvalues, proportion = prop.CCA, max = .1, shrinkage = .9)
  shared.matrix <- shared.covariance.generator.eigen(sparse.wishart.eigenvalues)
  view.noise <- noise.covariance.generator(data1.dim = D1, data2.dim = D2, shared.noise.1 = .3, shared.noise.2 = .5)
  generated.data <- view.specific.matrix %*% t(rmvnorm(n = n, mean = rep(0, ncol(view.specific.matrix)), sigma = diag(ncol(view.specific.matrix)))) + 
    shared.matrix%*%t(rmvnorm(n = n, mean = rep(0, ncol(shared.matrix)), sigma = diag(ncol(shared.matrix)))) + 
    t(rmvnorm(n = n, mean = rep(0, ncol(shared.matrix)), sigma = diag(ncol(shared.matrix)) + view.noise))
  
  CCA.list <- list(view.specific.column.variances, view.specific.matrix, wishart.eigenvalues,
                   sparse.wishart.eigenvalues, shared.matrix, generated.data)
  names(CCA.list) = c("view specific column variances", "view matrix", "original shared eigenvalues",
                      "sparse shared eigenvalues", "shared matrix", "generated data")
  return(CCA.list)
}
  
library(rstan)
library(extraDistr)
n = 1000
D1 = 5
D2 = 5
K1 = 2
K2 = 1
k = K1 + K2
d = D1 + D2

Generated.data <- CCA.normal.data.generator(n = n, D1 = D1, D2 = D2, K1 = K1, K2 = K2, prop.CCA = .7)

num.of.zero <- sum(Generated.data$`sparse shared eigenvalues`< .1)

CCA.data <- list(
  N = n,
  Y = t(Generated.data$'generated data'),
  D = d,
  D_1 = D1,
  D_2 = D2,
  K_1 = K1,
  K_2 = K2,
  Q = d,
  k = k
)

set.seed(4333213)
uni.max = max(Generated.data$`sparse shared eigenvalues`) + runif(1, min = 0, max = max(Generated.data$`sparse shared eigenvalues`)/10)
chain.1 = list(eigen_weight = sort(rbeta((d), shape1 = .1, shape2 = .1), decreasing = T),
               eigen_roots = sort(c(runif(d/2 , min = 0, max = 1),
                              uni.max*runif(d/2 , min = .25 , max = 1)), decreasing = T),
               column.variances = sort(rhcauchy(K1 + K2), decreasing = F))

chain.1 = list(column.variances = sort(rhcauchy(K1 + K2), decreasing = F))

names(chain.1) = c("column_variances")



fit.householder.desktop.1 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Fixed sparse householder CCA.stan", data = CCA.data, chains = 3, seed = 303, iter = 250, control = list(max_treedepth = 12, adapt_delta = .4))

fit.householder.desktop.1 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Fixed sparse householder CCA.stan", data = CCA.data, chains = 3, seed = 303, iter = 150, control = list(max_treedepth = 12, adapt_delta = .4))
fit.householder.desktop.2 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Fixed sparse householder CCA.stan", data = CCA.data, chains = 3, seed = 111011101, iter = 250, control = list(max_treedepth = 12, adapt_delta = .4))
fit.householder.desktop.3 <- stan(file = "C:/Users/qsimo/Documents/Code/RHouseholder/Fixed sparse householder CCA.stan", data = CCA.data, chains = 3, seed = 8, iter = 250)


estimates.1 <- get_posterior_mean(fit.householder.desktop.1, pars = c("eigen_roots"))
summary(fit.householder.desktop.1, pars = c("eigen_roots"))$summary[ , "50%"]
summary(fit.householder.desktop.2, pars = c("eigen_roots"))$summary[ , "50%"]
summary(fit.householder.desktop.3, pars = c("eigen_roots"))

summary(fit.householder.desktop.3, pars = c("column_variances"))$summary[ , "50%"]


dim.ratio <- function(CCA_data){
  return((CCA_data$Q)/CCA_data$N)
}

dim.ratio(CCA_data = CCA.data)

estimated.zeros <- function(fit){
  sum(summary(fit, pars = c("eigen_roots"))$summary[ , "50%"] < 1)
}

estimated.nonzeros <- function(fit){
  sum(summary(fit, pars = c("eigen_roots"))$summary[ , "50%"] >= 1)
}

posterior.mean.squared <- function(generated_data, fit){
  true.eigenvalues <- generated_data$`sparse shared eigenvalues`
  median.eigenvalues <- summary(fit, pars = c("eigen_roots"))$summary[ , "50%"]
  error <- rep(0,sum(Generated.data$`sparse shared eigenvalues` < 1))
  for(i in 1:length(error)){
    error[i] = true.eigenvalues[i] - median.eigenvalues[i]
  }
  return(mean(error^2))
}

true.zeros <- function(generated_data){
  return(sum(generated_data$`sparse shared eigenvalues` < 1))
}

true.nonzeros <- function(generated_data){
  return(sum(generated_data$`sparse shared eigenvalues` >= 1))
}

eigenvalue.proportion <- function(CCA_data){
  return(CCA_data$k/CCA_data$D)
}

eigenvalue.proportion(CCA_data = CCA.data)

true.zeros(generated_data = Generated.data)

true.nonzeros(generated_data = Generated.data)

estimated.zeros(fit = fit.householder.desktop.1)

estimated.zeros(fit = fit.householder.desktop.3)

posterior.mean.squared(generated_data = Generated.data,
                       fit = fit.householder.desktop.3)








stan_trace(fit.householder.desktop.1, pars = c("eigen_roots[1]","eigen_roots[2]", 
                                               "eigen_roots[3]", "eigen_roots[4]"))

stan_trace(fit.householder.desktop.2, pars = c("eigen_roots[13]","eigen_roots[14]", 
                                               "eigen_roots[15]", "eigen_roots[16]"))

stan_trace(fit.householder.desktop.3, pars = c("eigen_roots[1]","eigen_roots[2]", 
                                               "eigen_roots[3]", "eigen_roots[4]"))









summary(fit.householder.seed.2, pars = c("eigen_roots"))$summary
summary(fit.householder.seed.3, pars = c("eigen_differences"))$summary
summary(fit.householder.seed.3, pars = c("local_eigen_variance"))$summary
summary(fit.householder.seed.2, pars = c("eigen_variance"))$summary
summary(fit.householder, pars = c("weighted_eigen_variance"))$summary
summary(fit.householder.seed.1, pars = c("max_var"))$summary



plot(summary(fit.householder.desktop, pars = c("eigen_roots")))







for(q in 1:Q){
  eigen_differences_corrected[q] = eigen_differences[Q - q + 1];
}

eigen_roots[1] = exp(eigen_differences_corrected[1]);

{
  for(i in 2:Q){
    eigen_roots[i] = exp(eigen_differences_corrected[1] - sum(eigen_differences_corrected[2:i]));
  }
}

for(q in 1:Q){
  local_eigen_variance_corrected[q] = local_eigen_variance[Q - q + 1];
}

for(q in 1:Q){
  weighted_eigen_variance[q] = eigen_variance*local_eigen_variance[q];
}


for(q in 1:Q){
  local_eigen_variance[q] ~ beta(.95, .95);
}