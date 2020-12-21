Data <- read.csv("Precossed_Data.csv", header = T)
credits <- read.csv(file = "Data/tmdb_5000_credits.csv", header= T)

######################Genre

Data1 <- paste(Data$genre_1, Data$genre_2, Data$genre_3) 
Data1 <- as.data.frame(Data1)
# Tokenizing
genre_tokens <- tokens(Data1$Data1, what="word", remove_symbols = T, remove_numbers = T, remove_url = T)
# Lower Case
genre_tokens <- tokens_tolower(genre_tokens)

genre_tokens <- tokens_select(genre_tokens, stopwords(), selection = "remove")

# Stemming
genre_tokens <- tokens_wordstem(genre_tokens, language = "english")


# Unigrams
genre_tokens <- tokens_ngrams(genre_tokens, n = 1)

# Document Feature Matrix
genre_dfm <- dfm(genre_tokens, tolower = F)

# Matrix
genre_matrix <- as.matrix(genre_dfm)





#########TF IDF

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
genre.df <- apply(genre_matrix, 1, term.frequency)

# IDf 
genre.idf <-apply(genre_matrix, 2, inverse.doc.freq)

# TF_IDF
genre.tfidf <- apply(genre.df, 2, tf.idf,
                  idf = genre.idf)

genre.tfidf <- t(genre.tfidf)

# Removing INcomplete cases
incomplete.cases <- which(!complete.cases(genre.tfidf))
genre.tfidf[incomplete.cases, ] <- rep(0.0, ncol(genre.tfidf))
incomplete.cases <- which(!complete.cases(genre.tfidf))
genre.tfidf[incomplete.cases, ] <- rep(0.0, ncol(genre.tfidf))




# Method: 
# Dimensionality reduction using SVD
# Dimensionality Reduction

library(irlba)
gc()
registerDoSNOW(cl)
genre.irlba.1 <- irlba(t(genre.tfidf), nv = 10, maxit = 600)
genre.svd.1 <- data.frame(genre.irlba.1$v)





















# ACTORS
Data2 <- paste(Data$actor_1,"+", Data$actor_2, "+",Data$actor_3) 
Data2 <- as.data.frame(Data2)
# Tokenizing


# Tokenizing
AC_tokens <- tokens(Data2$Data2, what="word", remove_symbols = T, remove_numbers = T, remove_url = T)
# Lower Case
AC_tokens <- tokens_tolower(AC_tokens)

AC_tokens <- tokens_select(AC_tokens, stopwords(), selection = "remove")

# Stemming
AC_tokens <- tokens_wordstem(AC_tokens, language = "english")


# biigrams
AC_tokens <- tokens_ngrams(AC_tokens, n = 2)

# Document Feature Matrix
AC_dfm <- dfm(AC_tokens, tolower = F)

# Matrix
AC_matrix <- as.matrix(AC_dfm)



library(doSNOW)
gc()
cl <- makeCluster(7, type = "SOCK")
registerDoSNOW(cl)

# TF
AC.df <- apply(AC_matrix, 1, term.frequency)

# IDf 
AC.idf <-apply(AC_matrix, 2, inverse.doc.freq)

# TF_IDF
AC.tfidf <- apply(AC.df, 2, tf.idf,
                     idf = AC.idf)

AC.tfidf <- t(AC.tfidf)

# Removing INcomplete cases
incomplete.cases <- which(!complete.cases(AC.tfidf))
AC.tfidf[incomplete.cases, ] <- rep(0.0, ncol(AC.tfidf))
incomplete.cases <- which(!complete.cases(AC.tfidf))
AC.tfidf[incomplete.cases, ] <- rep(0.0, ncol(AC.tfidf))

library(irlba)
gc()
registerDoSNOW(cl)
AC.irlba.1 <- irlba(t(AC.tfidf), nv = 10, maxit = 600)
AC.svd.1 <- data.frame(AC.irlba.1$v)











################ Production Companies
# Production Companies
Data3 <- paste(Data$PDC1,"+", Data$PDC2) 
Data3 <- as.data.frame(Data3)
# Tokenizing


# Tokenizing
PDC_tokens <- tokens(Data3$Data3, what="word", remove_symbols = T, remove_numbers = T, remove_url = T)
# Lower Case
PDC_tokens <- tokens_tolower(PDC_tokens)

PDC_tokens <- tokens_select(PDC_tokens, stopwords(), selection = "remove")

# Stemming
PDC_tokens <- tokens_wordstem(PDC_tokens, language = "english")


# biigrams
PDC_tokens <- tokens_ngrams(PDC_tokens, n = 2)

# Document Feature Matrix
PDC_dfm <- dfm(PDC_tokens, tolower = F)

# Matrix
PDC_matrix <- as.matrix(PDC_dfm)



library(doSNOW)
gc()
cl <- makeCluster(7, type = "SOCK")
registerDoSNOW(cl)

# TF
PDC.df <- apply(PDC_matrix, 1, term.frequency)

# IDf 
PDC.idf <-apply(PDC_matrix, 2, inverse.doc.freq)

# TF_IDF
PDC.tfidf <- apply(PDC.df, 2, tf.idf,
                  idf = PDC.idf)

PDC.tfidf <- t(PDC.tfidf)

# Removing INcomplete cases
incomplete.cases <- which(!complete.cases(PDC.tfidf))
PDC.tfidf[incomplete.cases, ] <- rep(0.0, ncol(PDC.tfidf))
incomplete.cases <- which(!complete.cases(PDC.tfidf))
PDC.tfidf[incomplete.cases, ] <- rep(0.0, ncol(PDC.tfidf))

library(irlba)
gc()
registerDoSNOW(cl)
PDC.irlba.1 <- irlba(t(PDC.tfidf), nv = 10, maxit = 600)
PDC.svd.1 <- data.frame(PDC.irlba.1$v)


Data4 <- cbind(Data, genre.svd.1)
Data5 <- cbind(Data4, PDC.svd.1)
Data6 <- cbind(Data5, AC.svd.1)


# Triming Down the feature we don't need
Data6 <- Data6[, -c(30:129)]
Data6 <- Data6[,-c( 2, 3, 5, 7, 8, 10, 11, 15, 17, 21, 22, 23, 25, 26, 27, 28, 29)]
Data6 <- Data6[,-4]




#############################DIRECTOR
DR_tokens <- tokens(Data$director, what="word", remove_symbols = T, remove_numbers = T, remove_url = T)
# Lower Case
DR_tokens <- tokens_tolower(DR_tokens)

DR_tokens <- tokens_select(DR_tokens, stopwords(), selection = "remove")

# Stemming
DR_tokens <- tokens_wordstem(DR_tokens, language = "english")


# biigrams
DR_tokens <- tokens_ngrams(DR_tokens, n = 2)

# Document Feature Matrix
DR_dfm <- dfm(DR_tokens, tolower = F)

# Matrix
DR_matrix <- as.matrix(DR_dfm)



library(doSNOW)
gc()
cl <- makeCluster(7, type = "SOCK")
registerDoSNOW(cl)

# TF
DR.df <- apply(DR_matrix, 1, term.frequency)

# IDf 
DR.idf <-apply(DR_matrix, 2, inverse.doc.freq)

# TF_IDF
DR.tfidf <- apply(DR.df, 2, tf.idf,
                   idf = PDC.idf)

DR.tfidf <- t(DR.tfidf)

# Removing INcomplete cases
incomplete.cases <- which(!complete.cases(DR.tfidf))
DR.tfidf[incomplete.cases, ] <- rep(0.0, ncol(DR.tfidf))
incomplete.cases <- which(!complete.cases(DR.tfidf))
DR.tfidf[incomplete.cases, ] <- rep(0.0, ncol(DR.tfidf))

library(irlba)
gc()
registerDoSNOW(cl)
DR.irlba.1 <- irlba(t(DR.tfidf), nv = 10, maxit = 600)
DR.svd.1 <- data.frame(DR.irlba.1$v)


Data7 <- cbind(Data6, DR.svd.1)

write.csv(Data7, "Data_For_Rating.csv")






