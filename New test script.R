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




