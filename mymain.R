library(glmnet)
library(xgboost)

train = read.csv("train.csv")
train.x = train[, -c(1, 83)]
train.y = train[, 83]
train.x$Garage_Yr_Blt[is.na(train$Garage_Yr_Blt)] = 0
categorical.vars = colnames(train.x)[which(sapply(train.x, function(x) {
  mode(x) == "character"
}))]
train.matrix =
  as.matrix(train.x[, !(colnames(train.x) %in% categorical.vars), drop = FALSE])
for (var in categorical.vars) {
  mylevels = sort(unique(train.x[, var]))
  m = length(mylevels)
  m = ifelse(m > 2, m, 1)
  tmp.train = matrix(0, nrow(train.matrix), m)
  col.names = NULL
  for (j in 1:m) {
    tmp.train[train.x[, var] == mylevels[j], j] = 1
    col.names = c(col.names, paste(var, mylevels[j], sep = ""))
  }
  colnames(tmp.train) = col.names
  train.matrix = cbind(train.matrix, tmp.train)
}

test = read.csv("test.csv")
test.x = test[, -1]
test.x$Garage_Yr_Blt[is.na(test.x$Garage_Yr_Blt)] = 0
test.matrix =
  as.matrix(test.x[, !(colnames(test.x) %in% categorical.vars), drop = FALSE])
for (var in categorical.vars) {
  mylevels = sort(unique(train.x[, var]))
  m = length(mylevels)
  m = ifelse(m > 2, m, 1)
  tmp.test = matrix(0, nrow(test.matrix), m)
  col.names = NULL
  for (j in 1:m) {
    tmp.test[test.x[, var] == mylevels[j], j] = 1
    col.names = c(col.names, paste(var, mylevels[j], sep = ""))
  }
  colnames(tmp.test) = col.names
  test.matrix = cbind(test.matrix, tmp.test)
}

set.seed(0)
xgb.model = xgboost(
  train.matrix,
  log(train.y),
  eta = 1 / 32,
  subsample = 0.5,
  nrounds = 1024,
  verbose = FALSE
)
write.csv(data.frame("PID" = test[, 1],
                     "Sale_Price" =
                       exp(predict(xgb.model, newdata = test.matrix))),
          "mysubmission2.txt",
          row.names = FALSE)

for(var in c(
  "Lot_Frontage",
  "Lot_Area",
  "Mas_Vnr_Area",
  "BsmtFin_SF_2",
  "Bsmt_Unf_SF",
  "Total_Bsmt_SF",
  "First_Flr_SF",
  "Second_Flr_SF",
  "Gr_Liv_Area",
  "Garage_Area",
  "Wood_Deck_SF",
  "Open_Porch_SF",
  "Enclosed_Porch",
  "Three_season_porch",
  "Screen_Porch",
  "Misc_Val"
)) {
  tmp = train.matrix[, var]
  myquan = quantile(tmp, probs = 0.95, na.rm = TRUE)
  tmp[tmp > myquan] = myquan
  train.matrix[, var] = tmp
  tmp = test.matrix[, var]
  tmp[tmp > myquan] = myquan
  test.matrix[, var] = tmp
}
set.seed(0)
alpha.min = 0
cvm.min = min(cv.glmnet(train.matrix, log(train.y), alpha = 0)$cvm)
for (a in 1:4 / 4) {
  set.seed(0)
  cv.out = cv.glmnet(train.matrix, log(train.y), alpha = a)
  if (min(cv.out$cvm) < cvm.min) {
    cvm.min = min(cv.out$cvm)
    alpha.min = a
  }
}
set.seed(0)
cv.out = cv.glmnet(train.matrix, log(train.y), alpha = alpha.min)
write.csv(data.frame("PID" = test[, 1],
                     "Sale_Price" =
                       exp(as.vector(predict(cv.out, s = cv.out$lambda.min,
                                             newx = test.matrix)))),
          "mysubmission1.txt",
          row.names = FALSE)