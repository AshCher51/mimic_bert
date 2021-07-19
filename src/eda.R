library(tidyverse)

# read in data
data <- readr::read_csv('../data/temp.csv')
data <- data %>% select(-X1)
data %>% head()
data %>% View()

# structure of data
str(data)
dim(data)
count(data)

# visualizing imbalance among labels
labels <- data %>% select(-SUBJECT_ID, -TEXT) 
counts <- labels %>% colSums() %>% data.frame() 

counts$Name <- names(labels)
counts$Frequency <- counts$.

counts <- as_tibble(counts) %>% select(-.)

ggplot(counts, aes(x=Name, y=Frequency)) + 
  geom_col() +
  ggtitle('Frequency of Different Phenotypes among Patients in the MIMIC-III Database') + 
  theme(text = element_text(size=9.5),
        plot.title = element_text(hjust = 0.5))

# distribution of words per report
text <- data %>% select(TEXT)
text['LENGTH'] <- purrr::map(text$TEXT, nchar)

