library(rstan) #Lets you do STAN code
library(tidyverse) #Cleans up plots
library(RCurl) #Lets you pull data from a URL
library(ggfortify) #Lets ggplot know how to interpret PCA results
library(bayesplot)

#Reading the wisconsin data and putting it in a table
UCI_data_URL <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
names <- c('id_number', 'diagnosis', 'radius_mean', 
           'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 
           'concavity_mean','concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 
           'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 
           'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 
           'symmetry_worst', 'fractal_dimension_worst')
Y <- read.table(UCI_data_URL, sep = ',', col.names = names)
view(Y)


#Standardizing the numerical data
for(i in 1:32){
  if(is.numeric(Y[,i]) == TRUE){
    Y[,i] <- (Y[,i] - mean(Y[,i]))/sd(Y[,i])
  }
  else
  {Y[,i] <- Y[,i]}
}

#Keeping only numerical data
wisconsin.data <- as.matrix(Y[3:32])


#Generates p random columns between 1 and n without replacement
Column.generator <- function(n,p){
  v <- rep(0,10)
  v[1] <- rdunif(1,1,n)
  i = 1
  while(i <= p){
    x <- rdunif(1,1,n)
    while(any(x == v[1:i-1])){
      x <- rdunif(1,1,n)
    }
    v[i] <- x
    i <- i + 1
  }
  return(v)
}


pc.cols = Column.generator(ncol(wisconsin.data), 10)


#STAN code

file1 <- "D:/School/Projects/HouseholderMCMC/PPCA.stan"
file2 <- "D:/School/Projects/HouseholderMCMC/PPCA_House.stan"


wisconsin.data.stan <- list(
  Y = wisconsin.data,
  N = nrow(wisconsin.data),
  D = ncol(wisconsin.data),
  Q = 2
)

wisconsin.ppca.fit <- stan_model(file = file1)

wisconsin.ppca.samples <- sampling(wisconsin.ppca.fit, list(N = 569, D = 30, Q = 2, Y = wisconsin.data), iter = 2000, chains = 1, warmup = 1000)


params = rstan::extract(wisconsin.ppca.samples)

W.samples <- params$W

W.samples.pc.subset <- W.samples[1:1000, pc.cols, 1:2]

plot(W.samples[1:1000, pc.cols, 2] ~ W.samples[1:1000,pc.cols, 1], col = 1:10, xlab = "W1", ylab = "W2")

wisconsin.ppca.householder.fit <- stan_model(file = file2)

wisconsin.ppca.householder.sampler <- sampling(wisconsin.ppca.householder.fit , list(N = 569, D = 30, Q = 2, Y = wisconsin.data), iter = 2000, chains = 1, warmup = 1000)

W.ppca.householder.parameters <- rstan::extract(wisconsin.ppca.householder.sampler)

W.householder.samples <- W.ppca.householder.parameters$W

W.samples.pc.householder.subset <- W.householder.samples[1:1000, pc.cols, 1:2]

plot(W.householder.samples[1:1000, pc.cols, 2] ~ W.householder.samples[1:1000,pc.cols, 1], col = 1:10, xlab = "W1", ylab = "W2")
#Classical PCA solutions
wisc.prcomp.subset <- prcomp(x = wisconsin.data[,pc.cols], scale = TRUE)
PCAloadings <- wisc.prcomp.subset$rotation
plot(PCAloadings[,1:2], xlim = c(-2.5,2.5), ylim = c(-2.5,2.5), xlab = "W1", ylab = "W2")
points(W.samples[1:2000, pc.cols, 2] ~ W.samples[1:2000,pc.cols, 1], col = 1:10, xlab = "W1", ylab = "W2")











