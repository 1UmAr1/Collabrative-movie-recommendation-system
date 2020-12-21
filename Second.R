# Making the OverView of the movies a bit more usefull
Data <- read.csv(file = "Movies_Processed.csv", header = T)

library(quanteda)
# Separating overview from movies
overview <- Data$overview

overview <- as.data.frame(overview)

# Tokenizing
OV_tokens <- tokens(overview$overview, what="word", remove_symbols = T,
                    remove_separators = T)

# Lower Case
OV_tokens <- tokens_tolower(OV_tokens)

OV_tokens <- tokens_select(OV_tokens, stopwords(), selection = "remove")

# Stemming
OV_tokens <- tokens_wordstem(OV_tokens, language = "english")


# Unigrams
OV_tokens <- tokens_ngrams(OV_tokens, n = 1)

# Document Feature Matrix
OV_dfm <- dfm(OV_tokens, tolower = F)

# Matrix
OV_matrix <- as.matrix(OV_dfm)





term.frequency <- function(row){
  row / sum(row)
}
# Take the row and divide by the total of the row


# Function for calculating inverse document frequency
# We want to look at the columns . so col

inverse.doc.freq <- function(col){
  # Calculating documents in the columns
  corpus.size <- length(col)
  # Getting the number of rows where the column in not zero
  doc.count <- length(which(col > 0))
  log10(corpus.size / doc.count)
}

# Function for calcularing TF-IDF
tf.idf <- function(x, idf){
  x * idf
}




library(doSNOW)
gc()
cl <- makeCluster(7, type = "SOCK")
registerDoSNOW(cl)

# TF
OV.df <- apply(OV_matrix, 1, term.frequency)

# IDf 
OV.idf <-apply(OV_matrix, 2, inverse.doc.freq)

# TF_IDF
OV.tfidf <- apply(OV.df, 2, tf.idf,
                     idf = OV.idf)

OV.tfidf <- t(OV.tfidf)

# Removing INcomplete cases
incomplete.cases <- which(!complete.cases(OV.tfidf))
OV.tfidf[incomplete.cases, ] <- rep(0.0, ncol(OV.tfidf))
incomplete.cases <- which(!complete.cases(OV.tfidf))
OV.tfidf[incomplete.cases, ] <- rep(0.0, ncol(OV.tfidf))




# Method: 
# Dimensionality reduction using SVD
# Dimensionality Reduction

library(irlba)
gc()
registerDoSNOW(cl)
OV.irlba.1 <- irlba(t(OV.tfidf), nv = 100, maxit = 800)
OV.svd.1 <- data.frame(OV.irlba.1$v)

Data1 <- cbind(Data, OV.svd.1)
write.csv(Data1, "Svd_Processed_Data_set.csv")







  