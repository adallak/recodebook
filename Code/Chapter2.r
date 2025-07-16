####################################################################
### Section 2.5.1
####################################################################

## Import required packages
require(recode)
require(glmnet)

data(fat)

## specify the target and feature matrix
y = fat["brozek"]
X <- model.matrix(brozek ~  age + weight +
                        height + adipos +
                        neck + chest +
                        abdom + hip + thigh +
                        knee + ankle +
                        biceps + forearm +
                        wrist, data=fat)[,-1]

## Split to train and test datasets
set.seed(123)
sample <- sample(c(TRUE, FALSE), nrow(fat), replace=TRUE, prob=c(0.8,0.2))

y_train = y[sample,1]
X_train = X[sample,]

y_test = y[!sample,1]
X_test = X[!sample,]

#####################################################################
### The Least-Squares
#####################################################################

## Run linear regression
ls <- lm(brozek~age + weight +
            height + adipos +
            neck + chest +
            abdom + hip + thigh +
            knee + ankle +
            biceps + forearm +
            wrist, data=fat[sample,])
coef(ls)

## Predict and calculate MSE
yhat.ls <-  predict(ls, newdata = fat[!sample,])
mean((yhat.ls - y_test)^2)

#####################################################################
### The Ridge
#####################################################################

## Run ridge with 10-fold cross-validation
set.seed(123)
ridge.cv = cv.glmnet(X_train,y_train, alpha = 0)
coef(ridge.cv)

## Predict and calculate MSE
yhat.ridge <-  predict(ridge.cv, X_test)
mean((yhat.ridge - y_test)^2)

#####################################################################
### The Lasso
#####################################################################

## Run lasso
lasso.default <- glmnet(X_train,y_train, standardize = TRUE)
## Plot the solution path
plot(lasso.default)

## Predict for specific lambda = 0.5 and calculate MSE
yhat.default = predict(lasso.default, newx = X_test, 
                s = c(0.50))
mean((yhat.default - y_test)^2)

## Implement 10-fold cross-validation
set.seed(123)
lasso.cv = cv.glmnet(X_train,y_train)
## Plot the result
plot(lasso.cv)
lambda.cv = lasso.cv$lambda.min
print(lambda.cv)

## Predict and calculate MSE
yhat.cv <-  predict(lasso.cv, newx = X_test, s = lambda.cv)
mean((yhat_cv - y_test)^2)

#####################################################################
### The threhsolded lasso
#####################################################################

## Implement threshold-lasso with 10 fold cross-validation
thresh.lasso.cv = cv.threshlasso(X_train, y_train, 
                lambda = 0.01, min.thresh = 0.01, ngrid = 30, 
                nfold = 10, seed = 123)

thresh.lasso =  thresh.lasso.cv$thresh_min
print(thresh.lasso)

## Estimate MSE
yhat.thresh = predict.threshlasso(thresh.lasso.cv, X_test)
mean((yhat.thresh - y_test)^2)

#####################################################################
### The SCAD
#####################################################################

## Import packages
require(ncvreg)

## Implement SCAD with 10-fold cross-validation
scad.cv = cv.ncvreg(X_train, y_train, 
                  penalty = "SCAD", seed = 123)
plot(cv_scad)
lambda.scad = scad.cv$lambda.min
print(lambda.scad)

## Estimate MSE
yhat.scad = predict(scad.cv, X_test, lambda = lambda.scad)
mean((yhat.scad - y_test)^2)

#####################################################################
### The MCP
#####################################################################

## Implement MCP with 10-fold cross-validation
mcp.cv = cv.ncvreg(X_train, y_train,
                   penalty = "MCP", seed = 123)
plot(mcp.cv)
lambda.scad =  mcp.cv$lambda.min
print(lambda.scad)
[1] 0.04041433
## Estimate MSE
yhat.mcp = predict(mcp.cv, X_test, lambda = lambda.scad)
mean((yhat.mcp - y_test)^2) 

## print coefficients
coef(mcp.cv)

#####################################################################
### Error Variance Estimation
#####################################################################

require(natural)
# obtains standard deviation using OLS
sigma_ols = summary(yhat.ls)$sigma
print(sigma_ols)
# obtain standard deviation using organic lasso
olasso_res = olasso_cv(x = X_test, y = y_test,
                       intercept = TRUE, nfold = 3)
sigma_olasso = olasso_res$sig_obj
print(sigma_olasso)

# obtain standard deviation using natural lasso
nlasso_res = nlasso_cv(x = X_test, y = y_test,
                       intercept = TRUE, nfold = 3)
sigma_nlasso = nlasso_res$sig_obj
print(sigma_nlasso)

# obtain naive standard deviation
sigma_naive = nlasso_res$sig_naive
print(sigma_naive)

# obtain standard deviation using cross-validation
sigma_cv = nlasso_res$sig_df
print(sigma_cv)

#####################################################################
### Section 2.5.2 Regularization path
#####################################################################

require(recode)
## Simulation settings
p = 100
n.list = 2^c(7:12)
nsim = 100

## Define coefficients
beta = matrix(0, nrow = p, ncol = 1)
beta[1:4] = c(5,3,-3,-1)

bias.plot(n.list = n.list, beta = beta, nsim = nsim, 
    seed = 1234)

#####################################################################
### Section 2.5.3 Comparing Error Variance Estimators
#####################################################################

require(recode)
require(natural)
data(yearpredictionmsd)

set.seed(123)
index = sample.int(n = nrow(yearpredictionmsd), size = 9000, replace = F)
train = yearpredictionmsd[index,]
test = yearpredictionmsd[-index,]

#obtain standard deviation using ols
ols_result = lm(year~ ., data = train)
sigma_ols = summary(ols_result)$sigma

test_size = c(50, 100, 150)
n_replicate = 1000
# matrices to save the squared errors
sq.err_olasso = matrix(0, nrow = nsim, ncol = length(test_size))
sq.err_nlasso = matrix(0, nrow = nsim, ncol = length(test_size))
sq.err_naive  = matrix(0, nrow = nsim, ncol = length(test_size))
sq.err_cv     = matrix(0, nrow = nsim, ncol = length(test_size))

for (size in 1:length(test_size)){
  for (rep in 1:n_replicate){
    df = test[sample(test_size[size]),]
    x = df[,2:ncol(df)]
    y = df[,1]
    # obtain standard deviation using organic lass
    olasso_res = olasso_cv(x = as.matrix(x), y = y,
                           intercept = TRUE, nfold = 3)
    sigma_olasso = olasso_res$sig_obj
    # obtain standard deviation using natural lasso
    nlasso_res = nlasso_cv(x = as.matrix(x), y = y,
                           intercept = TRUE, nfold = 3)
    sigma_nlasso = nlasso_res$sig_obj
    # obtain naive standard deviation
    sigma_naive = nlasso_res$sig_naive
    # obtain standard deviation using cross-validation
    sigma_cv = nlasso_res$sig_df
    # estimate squared error
    sq.err_olasso[rep, size] = (sigma_olasso/sigma_ols - 1)^2
    sq.err_nlasso[rep, size] = (sigma_nlasso/sigma_ols - 1)^2
    sq.err_naive[rep, size] = (sigma_naive/sigma_ols - 1)^2
    sq.err_cv[rep, size] = (sigma_cv/sigma_ols - 1)^2
  }
}
# Compute mean squared error
mean_sq.err = rbind(colMeans(sq.err_olasso),
                     colMeans(sq.err_nlasso),
                     colMeans(sq.err_naive),
                     colMeans(sq.err_cv)) * 100

#####################################################################
### Section 2.5.4 Impact of Predictors Correlation on Model S
#####################################################################

require(MASS)
require(glmnet)
require(ncvreg)

## Parameters for data generation
n = 100
p = 200
nsim = 50

# List of correlations
rholist = seq(0,9)/10

# Covariance for the error term
Sigma_eps = 0.3 * diag(p)

methods <- c("Lasso", "SCAD", "MCP")
store_size = matrix(0, nrow = length(methods), ncol = length(rholist))
ind = 1

set.seed(123)

for (rho in rholist){
  Lasso_sel = SCAD_sel = MCP_sel = 0
  for (i in 1:nsim) {

    # Generate true beta
    beta_nz = runif(10, 2, 5)
    beta = c(beta_nz, rep(0,p - 10))

    # Generate correlated predictors
    Sigma_X = diag(p) * (1 - rho) + rho

    X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma_X)

    # Generate response
    y <-  X %*% beta + rnorm(n, 0, Sigma_eps)

    # Get model size
    ## Lasso
    lasso_fit <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
    lambda_selected = lasso_fit$lambda.min
    coef_lasso <- coef(lasso_fit, s = lambda_selected)[-1]
    Lasso_sel <- Lasso_sel + length(which(coef_lasso != 0))

    ## SCAD
    SCAD_fit = cv.ncvreg(X, y, nfolds = 10,
                      penalty = "SCAD")
    lambda_selected = SCAD_fit$lambda.min
    coef_SCAD <- coef(SCAD_fit, s = lambda_selected)[-1]
    SCAD_sel <- SCAD_sel + length(which(coef_SCAD != 0))

    ## MCP
    MCP_fit = cv.ncvreg(X, y, nfolds = 10,
                      penalty = "MCP")
    lambda_selected = MCP_fit$lambda.min
    coef_MCP <- coef(MCP_fit, s = lambda_selected)[-1]
    MCP_sel <- MCP_sel + length(which(coef_MCP != 0))
  }
  store_size[, ind] = c(Lasso_sel, SCAD_sel, MCP_sel) / nsim
  ind = ind + 1
}
