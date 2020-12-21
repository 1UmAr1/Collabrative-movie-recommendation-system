library(lsa)
Data <- read.csv(file = "Svd_Processed_Data_set.csv", header = T)


# Cosine Similarity
OV <- Data[,30:129]
OV.similarities <- cosine(t(as.matrix(OV)))


for(i in 1:nrow(OV)) {
  OV$OV.CosSim[i] <- mean(OV.similarities[i])
}




Data$OV.Cosim <- OV$OV.CosSim


# Slicing out the numerical features

num_features <- Data[,c(1, 9, 13, 14, 19, 20)]
num_features <- Data[,c(3, 11, 15, 16, 21, 22)]

# Checking for the corelation between numerical features.
library(corrplot)
corrplot(cor(num_features), 
         method = "number",
         type = "upper"
)
cor(Data$vote_average, Data$vote_count)
cor(Data$revenue, Data$budget)
cor(Data$popularity, Data$revenue)


write.csv(Data, "Precossed_Data.csv")







'''
# Jaccard Similarity

jaccard <- function(df, margin=1){
  if(margin ==1 | margin == 2){
    M_00 <- apply(df, margin, sum) == 0
    M_11 <- apply(df, margin, sum) == 2
    if (margin == 1){
      df <- df[!M_00,]
      JSim <- sum(M_11) / nrow(df)
    }else{
      df <- df[, !M_00]
      JSim <- sum(M_11) / length(df)
    }
    JDist <- 1 - JSim
    return(c(JSim = JSim, JDist = JDist))
  }else break
}




library(magrittr)
library(dplyr)
Jaccard_row <- function(df, margin=1){
  key_paris <- expand.grid(row.names(df), row.names(df))
  results <- t(apply(key_paris, 1, function(row) 
    jaccard(df[c(row[1], row[2]),], margin=margin)))
  key_pair <- key_paris %>% mutate(pair = paste(Var1, "_", Var2, sep=""))
  results <- data.frame(results)
  row.names(results) <- key_pair$pair
  results
}

jc <-Jaccard_row(OV[,-101], margin = 1)
'''

