# Loading the Data
movies <- read.csv(file = "Data/tmdb_5000_movies.csv", header = T)
credits <- read.csv(file = "Data/tmdb_5000_credits.csv", header= T)

# Data Analysis
summary(movies)
summary(credits)

head(movies)
summary(movies$vote_average)
dim(movies)

library(ggplot2)
ggplot(movies, aes(vote_average)) +
  geom_bar(fill = "blue") +
  xlab("Votes Average") + 
  theme_light()



# Focusing on the votes
range(movies$vote_count)

ggplot(movies, aes(x = vote_count)) + geom_histogram(fill="red") + 
  theme_minimal()+
  xlab("Number of Votes") +
  ylab("Number of Movies")





library(tidyverse)
library(jsonlite)

# Top Actors
cast <- credits %>%
  filter(nchar(cast) > 2) %>%
  mutate(js = lapply(cast, fromJSON)) %>%
  unnest(js) %>%
  select(-cast, -crew) %>%
  rename(actor=name, movies_cast_id=cast_id, actor_id=id) %>%
  mutate_if(is.character, factor)

cast1 <- cast %>% count(actor)
cast <- cast %>% filter(order %in% c(0, 1, 2)) %>% select(movie_id, title, order, actor)

cast$order[1] <- 0
for (i in 1:(nrow(cast) -1 )){
  if(cast$movie_id[i+1] != cast$movie_id[i]){
    cast$order[i + 1] <- 0
  }else {cast$order[i+ 1] <- cast$order[i] + 1}
}


cast <- cast %>% filter(order %in% c(0, 1, 2)) %>%
  spread(key = order, value = actor) %>%
  rename(actor_1="0", actor_2="1", actor_3="2")


movies <- left_join(movies, cast %>% select(id = movie_id, actor_1, actor_2, actor_3), by = "id")




# Popular Directors
# Directors play a vital role in making a movie great
all_crew <- credits %>%       
  filter(nchar(crew)>2) %>%         
  mutate(                          
    js  =  lapply(crew, fromJSON)  
  )  %>%                           
  unnest(js)    


director_wm<- all_crew %>%
  filter(job=="Director") %>%
  mutate(director=name) %>%
  left_join(
    movies,
    by=c("movie_id" = "id")
  ) 

# TOp Rated Directors
directors <- director_wm %>% 
  group_by(director) %>%
  filter(vote_count>10) %>%
  summarise(         
    n = n(),
    weighted_mean = 
      weighted.mean(vote_average, vote_count)) %>%
  filter(n>1) %>% 
  arrange(desc(weighted_mean)) 


# Visualizing
ggplot(directors) + stat_qq(aes(sample = weighted_mean))
ggplot(directors, aes(weighted_mean)) +
  geom_histogram() + 
  theme_bw()


# Most of the directors have movies voted above 6/10
crew <- credits %>%
  filter(nchar(crew)>2) %>%
  mutate(js = lapply(crew, fromJSON)) %>%
  unnest(js) %>%
  select(-cast, -crew, -credit_id) %>%
  rename(crew=name, crew_id=id) %>%
  mutate_if(is.character, factor)

movies1Director <- crew %>% filter(job=="Director") %>% count(movie_id) %>% filter(n==1)
movies <- left_join(movies, crew %>% filter(job=="Director" & movie_id %in%
                                              movies1Director$movie_id) %>% 
                      select(id=movie_id, director=crew), by = "id")




keywords <- movies %>%    
  filter(nchar(keywords)>2) %>%
  mutate(
    js = lapply(keywords, fromJSON)
  ) %>%       
  unnest(js, .name_repair = "unique") %>% 
  select(id, title, keyword=name) %>% 
  mutate_if(is.character, factor)



# Number of movies by genre
genres <- movies %>% filter(nchar(genres) > 2) %>%
  mutate(js = lapply(genres, fromJSON)) %>%
  unnest(js, .name_repair = "unique") %>%
  select(id, title, genres = name) %>%
  mutate_if(is.character, factor)


genres %>% group_by(genres) %>% count() %>%
  ggplot(aes(x=reorder(genres, n), y=n)) +
  geom_col(fill="navyblue") + coord_flip() +
  labs(x="", y="Number of movies")
  



# Keeping only top 3 genres
gen <- genres
gen$order <- 0
gen$order[1] <- 1

for(i in 1:(nrow(gen)-1)) {
  if(gen$id[i+1] !=gen$id[i]){
    gen$order[i+1] <- 1
  }else {gen$order[i+1] <- (gen$order[i]) + 1}
}

gen <- gen %>% filter(order < 4) %>%
  spread(key=order, value=genres) %>%
  rename(genre_1="1", genre_2="2", genre_3 = "3")

movies <- left_join(movies, gen %>% select(id, genre_1, genre_2, genre_3), by="id")


library(wordcloud)
# Keywords
keywords_counts <- keywords %>% count(keyword)
n_distinct(keywords$keyword)

keywords %>% count(keyword) %>% top_n(20, wt=n) %>%
  ggplot(aes(x=reorder(keyword, n), y=n)) +
  geom_col(fill="red") + coord_flip() +
  labs(x="", y="Number of movies")

# movies <- left_join(movies, keywords %>% select(keywords), by="id")
par(mfrow=c(1, 1),bg="white")
wordcloud(keywords_counts$keyword, keywords_counts$n, max.words = 100,
          scale=c(2.0,.5), random.color = TRUE,
          random.order=FALSE, rot.per=0, colors=brewer.pal(9,"Set1"))


# Production Companies
production_companies <- movies %>% filter(nchar(production_companies) > 5) %>%
  mutate(js = lapply(production_companies, fromJSON)) %>%
  unnest(js, .name_repair = "unique") %>%
  select(id, title, production_companies = name) %>%
  mutate_if(is.character, factor)

PDC <- production_companies
PDC$order <- 0
PDC$order[1] <- 1

for(i in 1:(nrow(PDC)-1)) {
  if(PDC$id[i+1] !=PDC$id[i]){
    PDC$order[i+1] <- 1
  }else {PDC$order[i+1] <- (PDC$order[i]) + 1}
}

PDC <- PDC %>% filter(order < 3) %>%
  spread(key=order, value=production_companies) %>%
  rename(PDC1="1", PDC2="2")

movies <- left_join(movies, PDC %>% select(id, PDC1, PDC2), by="id")

# movies <- movies[,-c(2, 3, 5, 10, 11, 15)]
write.csv(movies, "Movies_Processed.csv")


