
# Prediction
predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad 
  
  hidden.layer <- new.data %*% model$net.W1
  # neurons : logistic

  hidden.layer <- 1/(1+exp(-hidden.layer))
  
  result <- hidden.layer %*% t(model$net.W2)
}

# Train: build and train a 3-layers neural network 
train.dnn <- function(x, y, traindata=data, testdata=NULL,
                      model = NULL,
                      # set hidden layers and neurons
                      # currently, only support 1 hidden layer
                      hidden=3, 
                      # max iteration steps
                      maxit=20000,
                      # delta loss 
                      abstol=0.01,
                      # learning rate
                      lr = 0.01,
                      # regularization rate
                      reg = 0.001)
{
  
  # total number of training set
  N <- nrow(traindata)
  
  # extract the data and label
  # don't need atribute 
  X <- unname(data.matrix(traindata[,x]))
  
  Y <- traindata[,y]
  
  # create model or get model from parameter
  if(is.null(model)) {
    # number of input features
    D <- ncol(X)
    # number of categories for classification
    K <-1
    H <-hidden
    
    # create and init weights and bias 
    W1 <- 5*matrix(rnorm(D*H,sd=0.5), nrow=H, ncol=D)
    b1 <- matrix(rnorm(D,sd=0.5))
    net.W1<-cbind(W1,b1)
    
    W2 <- 5*matrix(rnorm(H*K,sd=0.5), nrow=K, ncol=H)
    b2 <- matrix(rnorm(K,sd=0.5))
    net.W2<-cbind(W2,b2)
  }
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # Training the network
  i <- 0
  for (i in 1:maxit) {
    
    # forward ....
    hidden.layer <- X %*% net.W1
    
    # neurons : logistic
    hidden.layer <- 1/(1+exp(-hidden.layer))
    
    result <- hidden.layer %*% t(net.W2)
    
    # compute the loss
    delta<-Y-result
    
    data.loss  <- sum(delta^2)
    
    reg.loss   <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
    
    loss<-rep(0,maxit)
    
    loss[i] <- data.loss + reg.loss
    
    i<-i+1
    
    if(loss[i]<-abstol)
      break
    
    # backward ....
    dresult<-delta
    
    dW2 <- t(hidden.layer) %*% dresult
    
    dhidden <- dresult %*% t(net.W2[,1:(length(net.W2[1,])-1)])*((hidden.layer)*(1-hidden.layer))
    
    dW1 <- t(X) %*% dhidden
    # update ....
    dW2 <- dW2 + reg*net.W2
    
    dW1 <- dW1  + reg*net.W1
    
    net.W1 <- net.W1 - lr * dW1
    
    net.W2 <- net.W2 - lr * dW2
  }
  
  # final results
  model <- list( D = D,
                 H = H,
                 K = K,
                 # weights and bias
                 net.W1= net.W1, 
                 net.W2= net.W2)
  
  return(model)
}

########################################################################
# testing
#######################################################################

getwd()

setwd('/Users/zhoudan/Neural Network')

A_index<-data.frame(read.csv("A_index.csv",header=T))

head(A_index)

A_index$X<-NULL

A_index$index<-NULL

head(A_index)

apply(A_index, 2, function(x) sum(is.na(x)))

dates<-as.Date(A_index[,1])

A_index$date<-NULL

nrow(A_index)

train<-A_index[c(0:500),]

test<-A_index[c(501:623),]

scaled<-as.data.frame(scale(A_index))

train_<-scaled[c(0:500),]

test_<-scaled[c(501:623),]

new_model<-train.dnn(x=1:3,y=4,traindata=train_,testdata=test_,hidden=3,maxit=20000)

predict<-predict.dnn(new_model,test_[,-4])

pr.nn_result<-predict*sd(A_index$close)+mean(A_index$close)

final<-data.frame(dates[501:623],test$close,pr.nn_result)

fig<-plot(test$close,pr.nn_result,ylab="Model predict value",xlab="Real value",col="blue",main="Real vs Predicted Neural Network",pch=18,cex=0.7)
abline(0,1,lwd=2)
legend("bottomright",legend="NN",pch=18,col="blue",bty="n")

